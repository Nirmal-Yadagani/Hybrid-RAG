import os
import json
from dotenv import load_dotenv

# MUST be loaded before DeepEval
load_dotenv()
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "120"
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval
)
# 🚨 IMPORT CHANGE: Swap OpenAIModel for GeminiModel
from deepeval.models import GeminiModel
from deepeval.evaluate import AsyncConfig

from retriever import QAbot

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    file_path='data/eval/goldens_100.json',
    input_key_name='input',
    expected_output_key_name='expected_output',
    context_key_name='context'
)

qa_agent = QAbot()
test_cases = []

print("Generating agent responses...")
for golden in dataset.goldens:
    result = qa_agent.ask_and_retrieve(golden.input)
    
    # Rebuild the rich context strings exactly as the LLM saw them!
    rich_contexts = []
    for doc in result['docs']:
        topic = doc.metadata.get("topic", "Unknown Topic")
        headings = doc.metadata.get("headings", "No Section")
        rich_chunk = (
            f"--- Source Topic: {topic} ---\n"
            f"Section: {headings}\n"
            f"{doc.page_content}"
        )
        rich_contexts.append(rich_chunk)
    
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(result['answer']),
        retrieval_context=rich_contexts, # <--- Pass the metadata-rich list here!
        expected_output=golden.expected_output
    )
    test_cases.append(test_case)

# 🚨 MODEL CHANGE: Initialize the Gemini judge
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


# 🚨 EXECUTION CHANGE: Chunking removed. The 4M TPM limit easily handles this.
print("\n--- Evaluating all 100 cases concurrently ---")

# Set a safe connection-pool concurrency. 
batch_config = AsyncConfig(max_concurrent=15)

# Evaluate returns a single TestRun object when not chunked
test_run = evaluate(
    test_cases, 
    retriever_metrics + generator_metrics,
    async_config=batch_config
    # 🚨 FIX: Removed ignore_errors=True
)

# Extract the standard results array from the TestRun object
results = test_run.test_results

# --- SAVE RAW CHECKPOINT ---
os.makedirs('data/eval', exist_ok=True)
raw_save_path = 'data/eval/raw_evaluations.json'

raw_data = []

# Manually extract the exact attributes to guarantee no class-method crashes
for res in results:
    metrics_list = []
    
    # Safely parse the metrics
    if hasattr(res, 'metrics_data'):
        for m in res.metrics_data:
            metrics_list.append({
                "name": getattr(m, "name", "Unknown"),
                "score": getattr(m, "score", 0.0),
                "is_successful": getattr(m, "is_successful", False),
                "reason": getattr(m, "reason", "No reason provided")
            })
            
    # Build the dictionary explicitly
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