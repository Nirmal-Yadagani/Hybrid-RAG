# export_results.py
import json
import os
import sys
import yaml
import argparse
import wandb

# Setup CLI Arguments
parser = argparse.ArgumentParser(description="Export evaluation results.")
parser.add_argument("--input", type=str, required=True, help="Path to the raw evaluations JSON")
parser.add_argument("--output", type=str, required=True, help="Path to save the final exported JSON")
parser.add_argument("--run_name", type=str, required=True, help="Name for the Weights & Biases run")
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import StageLogger  # noqa: E402
from wandb_config import common_kwargs  # noqa: E402

log = StageLogger("export")

raw_save_path = args.input
final_save_path = args.output

# 1. Load Config for W&B Hyperparameters
with open('params.yaml', 'r') as f:
    config_params = yaml.safe_load(f)

# 2. Initialize W&B Experiment Run
wandb.init(
    **common_kwargs(config_params),
    name=args.run_name,
    config=config_params
)

with open(raw_save_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

evaluation_export = []
metric_aggregates = {}

# 3. Create a W&B Table for row-by-row debugging
columns = ["Test Case", "Question", "Metric", "Score", "Pass", "Actual Answer", "Expected Answer", "Reason", "Context"]
wandb_table = wandb.Table(columns=columns)

for idx, test_result in enumerate(raw_data):
    input_text = test_result.get('input', 'N/A')
    actual_output = test_result.get('actual_output', 'N/A')
    expected_output = test_result.get('expected_output', 'N/A')
    retrieval_context = "\n---\n".join(test_result.get('retrieval_context', []))
    
    metrics_export = []
    metrics_data = test_result.get('metrics_data', [])
    
    for metric in metrics_data: 
        m_name = metric.get('name', 'Unknown')
        m_score = metric.get('score', 0)
        m_pass = bool(m_score >= 0.5)
        m_reason = metric.get('reason', 'No reason provided')
        
        # Track for local JSON aggregation
        if m_name not in metric_aggregates:
            metric_aggregates[m_name] = {"total_score": 0, "pass_count": 0, "total_count": 0}
        
        metric_aggregates[m_name]["total_score"] += m_score
        metric_aggregates[m_name]["total_count"] += 1
        if m_pass:
            metric_aggregates[m_name]["pass_count"] += 1
            
        metrics_export.append({
            "name": m_name, "score": m_score, "pass": m_pass, "reason": m_reason
        })

        # Add row to W&B Table
        wandb_table.add_data(
            f"TC-{idx+1}", input_text, m_name, m_score, m_pass, 
            actual_output, expected_output, m_reason, retrieval_context
        )
        
    evaluation_export.append({
        "input": input_text, "actual_output": actual_output,
        "expected_output": expected_output, "metrics": metrics_export
    })

# 4. Calculate Aggregate Summary and Log to W&B Dashboard
summary = {}
wandb_metrics = {}

for name, data in metric_aggregates.items():
    avg_score = data["total_score"] / data["total_count"] if data["total_count"] > 0 else 0
    pass_rate = (data["pass_count"] / data["total_count"]) * 100 if data["total_count"] > 0 else 0
    
    summary[name] = {
        "average_score": round(avg_score, 3),
        "pass_rate_percentage": round(pass_rate, 1)
    }
    
    # Flatten metrics for W&B charts (e.g., "Contextual Relevancy_avg_score")
    wandb_metrics[f"{name}_avg_score"] = round(avg_score, 3)
    wandb_metrics[f"{name}_pass_rate"] = round(pass_rate, 1)

# Log the flat metrics and the rich table
wandb.log(wandb_metrics)
wandb.log({"Detailed_Test_Cases": wandb_table})
wandb.finish()

final_payload = {
    "aggregate_summary": summary,
    "total_cases_evaluated": len(evaluation_export),
    "detailed_results": evaluation_export
}

os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
with open(final_save_path, 'w', encoding='utf-8') as f:
    json.dump(final_payload, f, indent=4)

print(f"\n✅ Formatted evaluation results saved & pushed to W&B!")
log.info(
    "export.done",
    "Results aggregated and exported",
    cases=len(evaluation_export),
    per_metric_avg={k: v["average_score"] for k, v in summary.items()},
)