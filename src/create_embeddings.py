import os
import sys
import time

# Stages run via DVC as `python src/<stage>.py`, which puts only ``src/`` on the
# path. Push the repo root (``src/data``'s parent) in so ``from src...`` imports
# resolve from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json

from dotenv import load_dotenv
from src.logger import StageLogger
from src.data import ChunkRecord, load_config, load_json

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

load_dotenv()

log = StageLogger("embed")


def load_docling_chunks(filepath: str) -> list[Document]:
    """Read the Docling chunked JSON dataset (validated as ``ChunkRecord`` objects)
    and convert them into LangChain ``Document`` objects."""
    langchain_docs = []
    for chunk in load_json(filepath, model=ChunkRecord):
        # Reconstruct the LangChain Document using the exact metadata we injected
        langchain_docs.append(
            Document(
                page_content=chunk.page_content,
                metadata=chunk.metadata.model_dump()
            )
        )

    print(f"Loaded {len(langchain_docs)} pre-chunked documents from JSON.")
    return langchain_docs


def create_embeddings(docs: list[Document], cfg):
    # Note: RecursiveCharacterTextSplitter is completely REMOVED.
    # The `docs` list already contains perfect Docling chunks!

    dense_model = cfg.dense_embedding_model
    sparse_model = cfg.sparse_embedding_model
    hf_token = os.getenv('HF_TOKEN')
    
    model_kwargs = {'device': 'cuda', 'token': hf_token}
    encode_kwargs = {'normalize_embeddings': cfg.normalize_embeddings}

    dense_embedder = HuggingFaceEmbeddings(
        model_name=dense_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    sparse_embedder = FastEmbedSparse(
        model_name=sparse_model,
        parallel=20,
        batch_size=256
    )

    # Initialize local in-memory client (or connect to a remote Qdrant instance)
    client = QdrantClient(path=cfg.persist_directory)
    collection_name = cfg.collection_name
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=cfg.dense_embd_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=cfg.collection_name,
        embedding=dense_embedder,
        sparse_embedding=sparse_embedder,
        retrieval_mode=RetrievalMode.HYBRID, 
        vector_name="dense",                    
        sparse_vector_name="sparse"
    )

    print('Creating dense and sparse embeddings for your Docling chunks...')
    log.info(
        "embed.start",
        "Embedding chunks into Qdrant",
        docs=len(docs),
        dense_model=cfg.dense_embedding_model,
        sparse_model=cfg.sparse_embedding_model,
        dim=cfg.dense_embd_dim,
        collection=cfg.collection_name,
    )
    embed_started = time.perf_counter()
    # Pass our pre-chunked docs directly to Qdrant
    vector_store.add_documents(documents=docs)
    print('Embeddings created and stored successfully!')

    # Write a small versionable status marker (the Qdrant store itself is a local
    # runtime artifact kept out of git/DVC) so experiments capture what was built.
    try:
        point_count = client.count(collection_name=cfg.collection_name).count
    except Exception:
        point_count = len(docs)
    status = {
        "collection": cfg.collection_name,
        "points": point_count,
        "dense": {"model": cfg.dense_embedding_model, "dim": cfg.dense_embd_dim},
        "sparse": {"model": cfg.sparse_embedding_model},
        "normalize_embeddings": cfg.normalize_embeddings,
        "source_chunks": "data/processed/ai_ml_rag_chunks.json",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open('data/embeddings_status.json', 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=4)

    log.info(
        "embed.done",
        "Embeddings created and stored",
        docs=len(docs),
        points=point_count,
        elapsed_s=round(time.perf_counter() - embed_started, 2),
    )


if __name__ == '__main__':
    from src.data import load_config

    cfg = load_config()

    # Path to the file created by the chunk stage
    input_file = "data/processed/ai_ml_rag_chunks.json"

    docs = load_docling_chunks(input_file)
    create_embeddings(docs=docs, cfg=cfg)