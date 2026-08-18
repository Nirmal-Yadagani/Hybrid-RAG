import os

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models


def create_embeddings(docs: list[Document]):
    load_dotenv()

    with open('config.yaml', 'r') as config:
        params = yaml.safe_load(config)

    splitter = RecursiveCharacterTextSplitter(chunk_size=params['chunk_size'], chunk_overlap=params['chunk_overlap'])
    
    splitted_docs = splitter.split_documents(documents=docs)

    dense_model = params['dense_embedding_model']
    sparse_model = params['sparse_embedding_model']
    hf_token = os.getenv('HF_TOKEN')
    model_kwargs = {'device': 'cuda',
                    'token': hf_token}
    encode_kwargs = {'normalize_embeddings': params['normalize_embeddings']}

    dense_embedder = HuggingFaceEmbeddings(model_name=dense_model,
                                     model_kwargs=model_kwargs,
                                     encode_kwargs=encode_kwargs)
    
    sparse_embedder = FastEmbedSparse(model_name=sparse_model,
                                      parallel=20,
                                      batch_size=256)

    # Initialize local in-memory client (or connect to a remote Qdrant instance)
    client = QdrantClient(path=params['persist_directory'])
    collection_name = params['collection_name']
    if not client.collection_exists(collection_name):
        client.create_collection(collection_name=collection_name,
                                 vectors_config={"dense": models.VectorParams(size=params['dense_embd_dim'], distance=models.Distance.COSINE)},
                                 sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))})

    vector_store = QdrantVectorStore(client=client,
                                     collection_name=params['collection_name'],
                                     embedding=dense_embedder,
                                     sparse_embedding=sparse_embedder,
                                     retrieval_mode=RetrievalMode.HYBRID, 
                                     vector_name="dense",                    
                                     sparse_vector_name="sparse")

    print('creating vector embeddings for your documents')
    vector_store.add_documents(documents=splitted_docs)
    print('embeddings created successfully')


if __name__=='__main__':

    from document_loader import load_docs

    docs = load_docs()
    embeddings = create_embeddings(docs=docs)

    