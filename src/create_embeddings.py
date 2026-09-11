import os
import json
import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models


def load_docling_chunks(filepath: str) -> list[Document]:
    """
    Reads the Docling chunked JSON dataset and converts it into LangChain Document objects.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    langchain_docs = []
    for chunk in raw_chunks:
        # Reconstruct the LangChain Document using the exact metadata we injected
        doc = Document(
            page_content=chunk["page_content"],
            metadata=chunk["metadata"]
        )
        langchain_docs.append(doc)
        
    print(f"Loaded {len(langchain_docs)} pre-chunked documents from JSON.")
    return langchain_docs


def create_embeddings(docs: list[Document]):
    load_dotenv()

    with open('config.yaml', 'r') as config:
        params = yaml.safe_load(config)

    # Note: RecursiveCharacterTextSplitter is completely REMOVED.
    # The `docs` list already contains perfect Docling chunks!

    dense_model = params['dense_embedding_model']
    sparse_model = params['sparse_embedding_model']
    hf_token = os.getenv('HF_TOKEN')
    
    model_kwargs = {'device': 'cuda', 'token': hf_token}
    encode_kwargs = {'normalize_embeddings': params['normalize_embeddings']}

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
    client = QdrantClient(path=params['persist_directory'])
    collection_name = params['collection_name']
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=params['dense_embd_dim'], distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=params['collection_name'],
        embedding=dense_embedder,
        sparse_embedding=sparse_embedder,
        retrieval_mode=RetrievalMode.HYBRID, 
        vector_name="dense",                    
        sparse_vector_name="sparse"
    )

    print('Creating dense and sparse embeddings for your Docling chunks...')
    # Pass our pre-chunked docs directly to Qdrant
    vector_store.add_documents(documents=docs)
    print('Embeddings created and stored successfully!')


if __name__ == '__main__':
    
    # Path to the file created by 2_chunk_data.py
    input_file = "data/processed/ai_ml_rag_chunks.json"
    
    docs = load_docling_chunks(input_file)
    create_embeddings(docs=docs)