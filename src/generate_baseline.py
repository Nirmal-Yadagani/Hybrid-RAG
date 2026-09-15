import os
import random
import sys
import time
import yaml
import wandb
import weave
from deepeval.dataset import EvaluationDataset
from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer
from retriever import QAbot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import StageLogger  # noqa: E402
from wandb_config import common_kwargs  # noqa: E402

log = StageLogger("generate")

with open('params.yaml', 'r') as config:
    params = yaml.safe_load(config)

# 1. Initialize Tracking & Tracing
# generator_model / temperature come from params.yaml so a dataset is traceable to the model that made it
generation_config = {
    "dataset_tier": "single-chunk-baseline",
    "generator_model": params["synthesizer_model"],
    "temperature": params["synthesizer_temperature"],
    "max_goldens_per_context": 1,
    "chunk_size_batch": 1,
    "min_word_count": 50,
    "target_dataset_size": 100
}

# wandb tracks the overall experiment and stores the dataset file
wandb.init(**common_kwargs(params), config=generation_config, name="generate_phase1_baseline")

# weave traces the latency, inputs, and outputs of the generation calls
weave.init("hybrid-rag-traces")

rag_bot = QAbot()

# 2. Fetch and Filter Chunks
records, _ = rag_bot.client.scroll(
    rag_bot.collection_name, 
    limit=1000, 
    with_payload=True, 
    with_vectors=False
)

valid_records = [
    r for r in records 
    if len(r.payload.get("page_content", "").split()) > generation_config["min_word_count"]
]
seed_points = random.sample(valid_records, min(generation_config["target_dataset_size"], len(valid_records)))

contexts = []
print(f"Preparing {len(seed_points)} isolated contexts for baseline generation...")

for point in seed_points:
    seed_text = point.payload.get("page_content")
    topic = point.payload.get("metadata", {}).get("topic", "Unknown Topic")
    heading = point.payload.get("metadata", {}).get("headings", "Unknown Section")
    
    rich_context = f"DOCUMENT TOPIC: {topic}\nSECTION: {heading}\nCONTENT:\n{seed_text}"
    contexts.append([rich_context])

# 3. Initialize Generator
ollama_llm = OllamaModel(
    model=generation_config["generator_model"], 
    base_url="http://localhost:11434",
    temperature=generation_config["temperature"] 
)
synthesizer = Synthesizer(model=ollama_llm)

# 4. Wrap generation in a Weave op to trace inputs/outputs and latency
@weave.op()
def generate_batch(context_chunk):
    return synthesizer.generate_goldens_from_contexts(
        contexts=context_chunk,
        include_expected_output=True,
        max_goldens_per_context=generation_config["max_goldens_per_context"]
    )

all_goldens = []
chunk_size = generation_config["chunk_size_batch"]
gen_started = time.perf_counter()

log.info(
    "generate.start",
    "Generating goldens",
    generator_model=generation_config["generator_model"],
    temperature=generation_config["temperature"],
    contexts=len(contexts),
    batch_size=chunk_size,
    min_word_count=generation_config["min_word_count"],
    target_size=generation_config["target_dataset_size"],
)
print(f"\nGenerating 100 goldens (1 per chunk) in batches of {chunk_size}...")
for i in range(0, len(contexts), chunk_size):
    batch_num = (i // chunk_size) + 1
    total_batches = (len(contexts) + chunk_size - 1) // chunk_size
    print(f"Processing batch {batch_num} of {total_batches}...")
    
    context_chunk = contexts[i : i + chunk_size]
    
    try:
        # Calls the traced function
        batch_goldens = generate_batch(context_chunk)
        all_goldens.extend(batch_goldens)
        
    except Exception as e:
        print(f"⚠️ Skipping context {i + 1} due to LLM failure: {e}")
        continue

# 5. Save locally
eval_dataset = EvaluationDataset(goldens=all_goldens)
os.makedirs('data/eval', exist_ok=True)
file_path = 'data/eval/goldens_100_baseline.json'
eval_dataset.save_as(
    file_name='goldens_100_baseline', 
    file_type='json', 
    directory='data/eval'
)

# 6. Version the dataset in W&B as an Artifact
artifact = wandb.Artifact(
    name="single-chunk-baseline-dataset",
    type="dataset",
    description="Phase 1: Isolated single chunks for baseline recall/precision testing."
)
artifact.add_file(file_path)
wandb.log_artifact(artifact)

wandb.finish()
rag_bot.client.close()
log.info(
    "generate.done",
    "Golden set generated",
    goldens=len(all_goldens),
    elapsed_s=round(time.perf_counter() - gen_started, 2),
)
print("\n✅ Single-Chunk Baseline dataset generated, traced via Weave, and versioned in W&B!")