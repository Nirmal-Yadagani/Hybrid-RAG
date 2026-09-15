import os
import json
import time
import sys
from dotenv import load_dotenv
from tqdm import tqdm

# MUST be loaded before DeepEval
load_dotenv()
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "120"
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import StageLogger  # noqa: E402

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval
)
from deepeval.models import GeminiModel
from deepeval.evaluate import AsyncConfig

log = StageLogger("eval")


# If the file exists, we load it instantly instead of generating again.
test_cases = []
cache_path = 'data/eval/generated_test_cases.json'

if os.path.exists(cache_path):
    print("⚡ Found cached agent responses! Skipping generation...")
    with open(cache_path, 'r', encoding='utf-8') as f:
        cached_data = json.load(f)
        for item in cached_data:
            test_cases.append(LLMTestCase(
                input=item['input'],
                actual_output=item['actual_output'],
                retrieval_context=item['retrieval_context'],
                expected_output=item['expected_output']
            ))
else:
    print("Generating agent responses...")
    from retriever import QAbot # Only import if we need to generate
    qa_agent = QAbot()
    
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(
        file_path='data/eval/goldens_100_baseline.json',
        input_key_name='input',
        expected_output_key_name='expected_output',
        context_key_name='context'
    )
    
    cache_export = []
    for golden in tqdm(dataset.goldens):
        result = qa_agent.ask_and_retrieve(golden.input)
        
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=str(result['answer']),
            retrieval_context=[doc.page_content for doc in result['docs']],
            expected_output=golden.expected_output
        )
        test_cases.append(test_case)
        
        # Build the cache payload
        cache_export.append({
            "input": golden.input,
            "actual_output": str(result['answer']),
            "retrieval_context": [doc.page_content for doc in result['docs']],
            "expected_output": golden.expected_output
        })
        
    os.makedirs('data/eval', exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_export, f, indent=4)
    print("✅ Agent responses successfully cached to disk!")

# --- INITIALIZE METRICS ---
local_judge = GeminiModel(
    model="gemini-3.5-flash-lite",
    api_key=os.getenv('GOOGLE_API_KEY'),
)

relevancy = ContextualRelevancyMetric(model=local_judge)
recall = ContextualRecallMetric(model=local_judge)
precision = ContextualPrecisionMetric(model=local_judge)

answer_correctness = GEval(
    name="Answer Correctness",
    criteria="Evaluate if the actual output's 'answer' property is correct and complete from the input and retrieved context. If the answer is not correct or complete, reduce score.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
    model=local_judge
)

citation_accuracy = GEval(
    name="Citation Accuracy",
    criteria="Check if the citations in the actual output are correct and relevant based on input and retrieved context. If they're not correct, reduce score.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
    model=local_judge
)

retriever_metrics = [relevancy, recall, precision]
generator_metrics = [answer_correctness, citation_accuracy]


# --- ADJUST CONCURRENCY FOR GOOGLE API STABILITY ---
print("\n--- Evaluating all 100 cases concurrently ---")

# Lowering this to ensures we don't trigger the Google GenAI SDK connection drops.
batch_config = AsyncConfig(max_concurrent=5)
log.info(
    "eval.start",
    "Running DeepEval suite",
    cases=len(test_cases),
    judge="gemini-3.5-flash-lite",
    max_concurrent=5,
)
eval_started = time.perf_counter()
test_run = evaluate(
    test_cases,
    retriever_metrics + generator_metrics,
    async_config=batch_config)
results = test_run.test_results
log.info(
    "eval.judged",
    "DeepEval finished judging",
    evaluated=len(results),
    elapsed_s=round(time.perf_counter() - eval_started, 2),
)

# --- SAVE RAW CHECKPOINT ---
os.makedirs('data/eval', exist_ok=True)
raw_save_path = 'data/eval/raw_evaluations.json'
raw_data = []

# Manually extract the exact attributes to guarantee no class-method crashes
for res in results:
    metrics_list = []
    if hasattr(res, 'metrics_data'):
        for m in res.metrics_data:
            metrics_list.append({
                "name": getattr(m, "name", "Unknown"),
                "score": getattr(m, "score", 0.0),
                "is_successful": getattr(m, "is_successful", False),
                "reason": getattr(m, "reason", "No reason provided")
            })
            
    raw_data.append({
        "input": getattr(res, "input", "N/A"),
        "actual_output": getattr(res, "actual_output", "N/A"),
        "expected_output": getattr(res, "expected_output", "N/A"),
        "retrieval_context": getattr(res, "retrieval_context", []),
        "metrics_data": metrics_list
    })

with open(raw_save_path, 'w', encoding='utf-8') as f:
    json.dump(raw_data, f, indent=4)

print(f"\n✅ Raw evaluation data safely saved to {raw_save_path}")
log.info("eval.done", "Raw evaluation saved to disk", saved=len(raw_data), path=raw_save_path)