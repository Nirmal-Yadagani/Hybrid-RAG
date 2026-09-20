import os
import random
import sys
import time

import pandas as pd
import weave
from deepeval.dataset import EvaluationDataset
from deepeval.models import OllamaModel
from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig, StylingConfig
from qdrant_client import QdrantClient

import wandb

# Run via `python src/<stage>.py`, so push the repo root first to make
# ``from src...`` imports resolve regardless of working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.logger import StageLogger
from src.data import load_config

log = StageLogger("generate")

cfg = load_config()

# 1. Initialize Configuration & Tracking
generation_config = {
    "dataset_tier": "single-chunk-baseline",
    "generator_model": cfg.synthesizer_model,
    "temperature": cfg.synthesizer_temperature,
    "min_word_count": 50,
    "target_dataset_size": 100,
    "quality_threshold": cfg.quality_threshold,
    "max_quality_retries": cfg.max_quality_retries,
    "min_golden_quality": cfg.min_golden_quality_to_keep,
    "num_evolutions": cfg.num_evolutions,
}

wandb.init(**cfg.wandb_common_kwargs(), config=cfg, name="generate_phase1_baseline")
weave.init("hybrid-rag-traces")

# 2. Fetch and Filter Chunks
db = QdrantClient(path=cfg.persist_directory)
records, _ = db.scroll(
    cfg.collection_name,
    limit=1000,
    with_payload=True,
    with_vectors=False
)

valid_records = [
    r for r in records
    if len(r.payload.get("page_content", "").split()) > generation_config["min_word_count"] # type: ignore
]
seed_points = random.sample(valid_records, min(generation_config["target_dataset_size"], len(valid_records)))

contexts = []
print(f"Preparing {len(seed_points)} isolated contexts for baseline generation...")
for point in seed_points:
    seed_text = point.payload.get("page_content") # type: ignore
    topic = point.payload.get("metadata", {}).get("topic", "Unknown Topic") # type: ignore
    heading = point.payload.get("metadata", {}).get("headings", "Unknown Section") # type: ignore
    contexts.append([f"DOCUMENT TOPIC: {topic}\nSECTION: {heading}\nCONTENT:\n{seed_text}"])

db.close() # Free up resources early

# 3. Initialize Generator
ollama_llm = OllamaModel(
    model=generation_config["generator_model"],
    base_url="http://localhost:11434",
    temperature=generation_config["temperature"]
)

filtration_config = FiltrationConfig(
    critic_model=ollama_llm,
    synthetic_input_quality_threshold=generation_config["quality_threshold"],
    max_quality_retries=generation_config["max_quality_retries"],
)

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.CONCRETIZING: 0.4,
        Evolution.CONSTRAINED: 0.3,
        Evolution.REASONING: 0.3,
    },
    num_evolutions=generation_config["num_evolutions"],
)

styling_config = StylingConfig(
    task="Machine Learning and Artificial Intelligence concepts",
    scenario="A software engineer or student studying AI",
    input_format="Natural language questions a real user would type",
    expected_output_format="A direct, factual answer grounded only in the given content",
)

synthesizer = Synthesizer(
    model=ollama_llm,
    filtration_config=filtration_config,
    evolution_config=evolution_config,
    styling_config=styling_config,
)

# 4. Wrap generation in a Weave op to trace inputs/outputs and latency
@weave.op()
def generate_single_context(ctx_list):
    synthesizer.generate_goldens_from_contexts(
        contexts=[ctx_list],
        include_expected_output=True,
        max_goldens_per_context=1
    )
    # Extract the internal state immediately before the next loop wipes it
    return synthesizer.synthetic_goldens, synthesizer.to_pandas()

gen_started = time.perf_counter()
log.info("generate.start", "Generating goldens", **generation_config)

all_goldens = []
all_dfs = []

print(f"\nGenerating {len(contexts)} goldens (1 per chunk)...")
for i, ctx in enumerate(contexts, 1):
    print(f"Processing context {i} of {len(contexts)}...")
    try:
        batch_goldens, batch_df = generate_single_context(ctx)
        all_goldens.extend(batch_goldens)
        all_dfs.append(batch_df)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Skipping context {i} due to LLM failure: {e}")
        continue

# 5. Quality report + hard filter
# Rebuild the master DataFrame and Goldens list safely
if all_dfs:
    df = pd.concat(all_dfs, ignore_index=True)
else:
    df = pd.DataFrame()

pre_filter_count = len(all_goldens)

if not df.empty and "synthetic_input_quality" in df.columns:
    mean_q = float(df["synthetic_input_quality"].mean())
    min_q = float(df["synthetic_input_quality"].min())
    below_threshold = int((df["synthetic_input_quality"] < generation_config["quality_threshold"]).sum())

    log.info(  # noqa: PLE1205
        "generate.quality",
        "Golden quality distribution",
        mean_input_quality=round(mean_q, 3),
        min_input_quality=round(min_q, 3),
        below_retry_threshold=below_threshold,
    )
    wandb.log({
        "quality/mean_input_quality": mean_q,
        "quality/min_input_quality": min_q,
        "quality/pct_below_threshold": below_threshold / len(df),
    })

    # Hard floor: drop goldens still below this score even after retries.
    keep_mask = df["synthetic_input_quality"] >= generation_config["min_golden_quality"]
    final_goldens = [g for g, keep in zip(all_goldens, keep_mask) if keep]
    
    dropped = pre_filter_count - len(final_goldens)
    if dropped:
        print(f"Dropped {dropped} goldens below quality floor {generation_config['min_golden_quality']}")
else:
    print("⚠️ No quality scores available — skipping quality filter.")
    final_goldens = all_goldens

# 6. Save locally
os.makedirs('data/eval', exist_ok=True)
file_path = 'data/eval/goldens_100_raw.json'
eval_dataset = EvaluationDataset(goldens=final_goldens)
eval_dataset.save_as(file_name='goldens_100_raw', file_type='json', directory='data/eval')

# 7. Version the dataset in W&B as an Artifact
artifact = wandb.Artifact(
    name="single-chunk-baseline-dataset",
    type="dataset",
    description="Phase 1: Isolated single chunks for baseline recall/precision testing."
)
artifact.add_file(file_path)
wandb.log_artifact(artifact)

wandb.finish()
log.info(  # noqa: PLE1205
    "generate.done",
    "Golden set generated",
    goldens=len(final_goldens),
    dropped_low_quality=pre_filter_count - len(final_goldens),
    elapsed_s=round(time.perf_counter() - gen_started, 2),
)
print(f"\n✅ {len(final_goldens)} goldens kept (of {pre_filter_count} generated), traced via Weave, and versioned in W&B!")