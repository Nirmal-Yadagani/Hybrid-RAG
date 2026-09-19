import json
import os
import re
import sys
import time
import yaml
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import StageLogger  # noqa: E402

log = StageLogger("validate_goldens")

# 1. Load Parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

validator_model = params.get("validator_model", "gemma4:31b")
validator_temperature = params.get("validator_temperature", 0.0)

INPUT_PATH = "data/eval/goldens_100_raw.json"
OUTPUT_PATH = "data/eval/goldens_clean.json"

# 2. Tier 1: Deterministic Heuristics
# Catches scraping junk, flattened tables, and reference lists before calling the LLM
BANNED_PATTERNS = [
    r"\b(cite_note|cite_ref|reflist|isbn|doi|pmid|bibcode|arxiv)\b",
    r"\b(external links?|further reading|see also|notes and references)\b",
    r"\b(as listed above|in the table below|according to the text above)\b",
    r"\[\s*\d+\s*\]",                # Stray reference numbers like [1], [44]
    r"(\. ){3,}",                    # Docling table-flattening artifacts (e.g. ". . . .")
    r"Compute capability \(version\)" # Unformatted CUDA capability grids
]

def passes_heuristics(question: str, expected_output: str) -> tuple[bool, str]:
    combined = f"{question} {expected_output}".lower()
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, combined):
            return False, f"Matched banned pattern: '{pattern}'"
    
    if len(question.strip().split()) < 5:
        return False, "Question too short (< 5 words)"
    
    if len(expected_output.strip().split()) < 4:
        return False, "Expected output too short (< 4 words)"
        
    return True, "OK"

# 3. Tier 2: LLM-as-a-Judge Faithfulness & Sufficiency Critic
class GoldenAudit(BaseModel):
    is_answerable: bool = Field(
        description="True if the question can be completely and accurately answered using ONLY the facts explicitly stated in the context."
    )
    is_faithful: bool = Field(
        description="True if the expected answer relies EXCLUSIVELY on facts present in the context. False if the answer includes outside domain knowledge, unstated implications, or external facts."
    )
    critique: str = Field(
        description="Concise rationale explaining the evaluation verdict."
    )

critic_llm = ChatOllama(
    model=validator_model,
    temperature=validator_temperature,
    base_url="http://localhost:11434"
).with_structured_output(GoldenAudit)

critic_prompt = ChatPromptTemplate.from_template("""You are an expert benchmark auditor validating test cases for a RAG evaluation system.

Context Chunk:
\"\"\"{context}\"\"\"

Question:
\"\"\"{question}\"\"\"

Expected Answer:
\"\"\"{expected_output}\"\"\"

Audit the test case against two criteria:
1. is_answerable: Can someone fully answer the question using ONLY the provided context chunk? If the question asks for details not mentioned in the chunk, mark FALSE.
2. is_faithful: Is the expected answer 100% grounded in the chunk? If the expected answer brings in outside facts, technical knowledge, or explanations that DO NOT appear in the chunk text, mark FALSE.
""")

critic_chain = critic_prompt | critic_llm

def validate_dataset():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file {INPUT_PATH} does not exist. Run the generate stage first.")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # DeepEval saves datasets either as a list of dicts or {"goldens": [...]}
    goldens = raw_data.get("goldens", raw_data) if isinstance(raw_data, dict) else raw_data

    clean_goldens = []
    dropped_heuristic = 0
    dropped_critic = 0
    started = time.perf_counter()

    log.info(
        "validate.start",
        "Starting golden validation",
        total_input=len(goldens),
        model=validator_model,
        temperature=validator_temperature
    )
    print(f"\nAuditing {len(goldens)} raw goldens with {validator_model}...")

    for idx, item in enumerate(goldens):
        question = item.get("input", "")
        expected_output = item.get("expected_output", "")
        
        raw_context = item.get("context", [])
        context_str = "\n".join(raw_context) if isinstance(raw_context, list) else str(raw_context)

        # Step 1: Heuristic filter
        passed_h, reason = passes_heuristics(question, expected_output)
        if not passed_h:
            print(f"❌ [TC-{idx+1}] Dropped by Heuristics: {reason}")
            dropped_heuristic += 1
            continue

        # Step 2: LLM Critic audit
        try:
            audit: GoldenAudit = critic_chain.invoke({
                "context": context_str,
                "question": question,
                "expected_output": expected_output
            })

            if audit.is_answerable and audit.is_faithful:
                clean_goldens.append(item)
                print(f"✅ [TC-{idx+1}] Passed audit")
            else:
                failure_reasons = []
                if not audit.is_answerable:
                    failure_reasons.append("Unanswerable from context")
                if not audit.is_faithful:
                    failure_reasons.append("External hallucination in answer")
                
                reason_msg = " & ".join(failure_reasons)
                print(f"⚠️  [TC-{idx+1}] Dropped by Critic ({reason_msg}): {audit.critique}")
                dropped_critic += 1

        except Exception as e:
            print(f"⚠️  [TC-{idx+1}] Critic execution failed: {e}. Dropping for safety.")
            dropped_critic += 1

    elapsed = round(time.perf_counter() - started, 2)
    retained = len(clean_goldens)
    
    print(f"\n{'='*70}")
    print(f"Validation Summary:")
    print(f"  Input Count        : {len(goldens)}")
    print(f"  Dropped (Heuristics): {dropped_heuristic}")
    print(f"  Dropped (Critic)   : {dropped_critic}")
    print(f"  Retained Clean     : {retained} ({round(retained / len(goldens) * 100, 1)}%)")
    print(f"  Time Elapsed       : {elapsed}s")
    print(f"{'='*70}\n")

    log.info(
        "validate.done",
        "Validation complete",
        retained=retained,
        dropped_heuristic=dropped_heuristic,
        dropped_critic=dropped_critic,
        elapsed_s=elapsed
    )

    # Save output in identical DeepEval-compatible structure
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_goldens, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    validate_dataset()