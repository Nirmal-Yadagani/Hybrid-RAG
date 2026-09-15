# 🤖 Hybrid-RAG Wikipedia Assistant

An advanced, production-ready Retrieval-Augmented Generation (RAG) pipeline built to scrape, chunk, index, and query Wikipedia articles. 

This project goes beyond simple vector search by implementing **Hybrid Retrieval** (Dense + Sparse embeddings) backed by **Cross-Encoder Reranking**, and includes a complete LLM-as-a-Judge evaluation suite using DeepEval and Streamlit.

## ✨ Features

* **Advanced Retrieval Strategy**: Combines Dense vectors (`BAAI/bge-base-en-v1.5`) and Sparse vectors (`Qdrant/bm25`) using a local Qdrant vector database.
* **Precision Reranking**: Re-scores top retrieved documents using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to ensure maximum contextual relevance.
* **Intelligent Document Processing**: Utilizes `scrapling` for stealthy extraction and `docling` for smart, hierarchical HTML chunking that preserves structural metadata.
* **Interactive Terminal UI**: A beautiful, rich-text command-line interface for interacting with the local `gemma4:31b` model.
* **Automated Evaluation Suite**: Generates synthetic "golden" test cases via Ollama, evaluates them using Google Gemini (`gemini-3.5-flash-lite`) via DeepEval, and visualizes the metrics in a Streamlit dashboard.

## 🛠️ Tech Stack

* **Framework**: LangChain, LangChain-Core
* **Local LLM**: Ollama (`gemma4:31b` for chat, `qwen3.6:latest` for synthetic data)
* **Vector Store**: Qdrant (Local via `qdrant-client`)
* **Embeddings & Reranking**: HuggingFace, FastEmbed, SentenceTransformers
* **Evaluation**: DeepEval, Streamlit, Plotly
* **Package Management**: `uv`

---

## ⚙️ Prerequisites

1. **Python 3.12+**
2. **[Ollama](https://ollama.com/)** installed and running locally. Pull the required models:
   ```bash
   ollama pull gemma4:31b
   ollama pull qwen3.6:latest
   ```
3. **API Keys**: You will need a HuggingFace Token (for embedding models) and a Google Gemini API Key (for the DeepEval judge).

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd hybrid-rag
   ```

2. **Install dependencies using `uv`:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your keys:
   ```env
   HF_TOKEN=your_huggingface_token
   GOOGLE_API_KEY=your_gemini_api_key
   ```

4. **Prepare Target URLs:**
   Ensure you have a text file containing Wikipedia URLs to scrape at `data/urls/database_urls.txt` (as defined in `params.yaml`).

---

## 📖 Usage Guide

The project is broken down into modular steps. Run them sequentially from the root directory.

### Phase 1: Data Pipeline
1. **Scrape Wikipedia Articles:**
   Extracts clean HTML and global metadata.
   ```bash
   python src/create_dataset.py
   ```
2. **Chunk Documents:**
   Processes the raw HTML into semantically aware chunks using Docling.
   ```bash
   python src/create_chunks.py
   ```
3. **Create Embeddings:**
   Generates dense and sparse embeddings and loads them into the local Qdrant collection.
   ```bash
   python src/create_embeddings.py
   ```

### Phase 2: Chat
Launch the rich terminal interface to chat with your localized Wikipedia assistant:
```bash
python main.py
```
*(Type `exit` or `quit` to end the session).*

### Phase 3: Evaluation & Metrics
This project takes LLM reliability seriously. The evaluation pipeline assesses Contextual Relevancy, Recall, Precision, Answer Correctness, and Citation Accuracy.

1. **Generate Synthetic Test Cases:**
   Uses your context chunks and `qwen3.6` to generate 100 Q&A pairs.
   ```bash
   python src/create_eval_dataset.py
   ```
2. **Run DeepEval Suite:**
   Evaluates the RAG pipeline using Gemini as the LLM Judge.
   ```bash
   python src/run_evals.py
   ```
3. **Process Results:**
   Formats the raw evaluation data into an aggregated summary.
   ```bash
   python src/export_results.py
   ```
4. **Launch the Dashboard:**
   View a detailed, interactive visualization of your pipeline's performance.
   ```bash
   streamlit run src/metrics_dashboard.py
   ```

## 🎛️ Configuration
You can tweak the models, chunking behaviors, and retrieval parameters directly in `params.yaml`:
* `dense_embedding_model` / `sparse_embedding_model`
* `rerank_model`
* `top_n` (Number of documents to send to the generator after reranking)