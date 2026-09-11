import json
import os

raw_save_path = 'data/eval/raw_evaluations.json'
final_save_path = 'data/eval/evaluation_results.json'

# Load the raw checkpoint data
with open(raw_save_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

evaluation_export = []
metric_aggregates = {}

for idx, test_result in enumerate(raw_data):
    input_text = test_result.get('input', 'N/A')
    actual_output = test_result.get('actual_output', 'N/A')
    expected_output = test_result.get('expected_output', 'N/A')
    retrieval_context = test_result.get('retrieval_context', [])
    
    print(f"\n================ Test Case {idx + 1} ================")
    print(f"Question: {input_text}")
    
    metrics_export = []
    metrics_data = test_result.get('metrics_data', [])
    
    for metric in metrics_data: 
        m_name = metric.get('name', 'Unknown')
        m_score = metric.get('score', 0)
        m_pass = metric.get('is_successful', False)
        m_reason = metric.get('reason', 'No reason provided')
        
        print(f" -> {m_name} Score: {m_score} (Pass: {m_pass})")
        print(f"    Reason: {m_reason}")
        
        # Track for aggregation
        if m_name not in metric_aggregates:
            metric_aggregates[m_name] = {"total_score": 0, "pass_count": 0, "total_count": 0}
        
        metric_aggregates[m_name]["total_score"] += m_score
        metric_aggregates[m_name]["total_count"] += 1
        if m_pass:
            metric_aggregates[m_name]["pass_count"] += 1
        
        metrics_export.append({
            "name": m_name,
            "score": m_score,
            "pass": m_pass,
            "reason": m_reason
        })
        
    evaluation_export.append({
        "input": input_text,
        "actual_output": actual_output,
        "expected_output": expected_output,
        "retrieval_context": retrieval_context,
        "metrics": metrics_export
    })

# --- NEW: Calculate Aggregate Summary ---
summary = {}
for name, data in metric_aggregates.items():
    avg_score = data["total_score"] / data["total_count"] if data["total_count"] > 0 else 0
    pass_rate = (data["pass_count"] / data["total_count"]) * 100 if data["total_count"] > 0 else 0
    summary[name] = {
        "average_score": round(avg_score, 3),
        "pass_rate_percentage": round(pass_rate, 1)
    }

# Wrap the final output to include the summary at the top
final_payload = {
    "aggregate_summary": summary,
    "total_cases_evaluated": len(evaluation_export),
    "detailed_results": evaluation_export
}

# Save the beautifully formatted output
os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
with open(final_save_path, 'w', encoding='utf-8') as f:
    json.dump(final_payload, f, indent=4)

print(f"\n✅ Formatted evaluation results successfully saved to {final_save_path}!")
print("\n--- Aggregate Summary ---")
for m_name, stats in summary.items():
    print(f"{m_name}: Avg Score {stats['average_score']} | Pass Rate {stats['pass_rate_percentage']}%")