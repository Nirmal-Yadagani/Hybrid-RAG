import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide", page_title="RAG Evaluation Dashboard")

# 1. Load the custom JSON data
@st.cache_data
def load_data():
    with open('data/eval/evaluation_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    data = load_data()
except FileNotFoundError:
    st.error("Could not find data/eval/evaluation_results.json. Run the evaluator first.")
    st.stop()

# --- FIX: Handle the new JSON structure ---
summary_data = data.get("aggregate_summary", {})
detailed_results = data.get("detailed_results", [])

# 2. Flatten the detailed data for Pandas and Plotly
flattened_metrics = []
for idx, test in enumerate(detailed_results):
    for metric in test.get('metrics', []):
        flattened_metrics.append({
            "Test ID": f"TC-{idx+1}",
            "Question": test.get('input'),
            "Actual Output": test.get('actual_output'),
            "Expected Output": test.get('expected_output'),
            "Metric": metric.get('name'),
            "Score": metric.get('score'),
            "Pass": metric.get('pass'),
            "Reason": metric.get('reason')
        })

df = pd.DataFrame(flattened_metrics)

# 3. Build the UI
st.title("🎯 Hybrid-RAG Evaluation Dashboard")

# --- NEW: Top-Level KPI Cards ---
st.subheader("Executive Summary")
# Create a column for every metric in the summary dynamically
kpi_cols = st.columns(len(summary_data))
for i, (metric_name, stats) in enumerate(summary_data.items()):
    with kpi_cols[i]:
        # Displays the Pass Rate prominently, with the Avg Score below it
        st.metric(
            label=f"{metric_name} (Pass Rate)", 
            value=f"{stats['pass_rate_percentage']}%", 
            delta=f"Avg Score: {stats['average_score']}",
            delta_color="off" # Keeps it gray instead of green/red since it's just an info point
        )

st.divider()

# Aggregate Metrics Row
col1, col2 = st.columns(2)
with col1:
    st.subheader("Average Score per Metric")
    avg_scores = df.groupby('Metric')['Score'].mean().reset_index()
    fig1 = px.bar(avg_scores, x='Metric', y='Score', text_auto='.2f', color='Metric')
    fig1.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Pass/Fail Rate per Metric")
    pass_rates = df.groupby(['Metric', 'Pass']).size().reset_index(name='Count')
    fig2 = px.bar(pass_rates, x='Metric', y='Count', color='Pass', barmode='stack', 
                  color_discrete_map={True: '#28a745', False: '#dc3545'})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Interactive Explorer
st.subheader("🔍 Deep Dive: Debug Failed Metrics")

# Filter by Metric and Pass/Fail
selected_metric = st.selectbox("Filter by Metric", df['Metric'].unique())
show_only_failures = st.checkbox("Show ONLY Failures", value=True)

filtered_df = df[df['Metric'] == selected_metric]
if show_only_failures:
    filtered_df = filtered_df[filtered_df['Pass'] == False]

# Display the reasons
if filtered_df.empty:
    st.success(f"No failures found for {selected_metric}! 🎉")
else:
    for _, row in filtered_df.iterrows():
        # Using status emojis to make it easy to read
        status_icon = "❌" if not row['Pass'] else "✅"
        with st.expander(f"{status_icon} {row['Test ID']} (Score: {row['Score']}) - {row['Question']}"):
            st.markdown(f"**Expected Answer:** {row['Expected Output']}")
            st.markdown(f"**Agent Answer:** {row['Actual Output']}")
            st.markdown("---")
            st.markdown(f"**Judge Reason:**\n> {row['Reason']}")