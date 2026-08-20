import os

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder


class QAbot:
    def __init__(self):
        with open('config.yaml', 'r') as config:
            params = yaml.safe_load(config)

        dense_model = params['dense_embedding_model']
        sparse_model = params['sparse_embedding_model']
        hf_token = os.getenv('HF_TOKEN')
        model_kwargs = {'device': 'cuda',
                        'token': hf_token}
        encode_kwargs = {'normalize_embeddings': params['normalize_embeddings']}
        self.client = QdrantClient(path=params['persist_directory'])
        self.collection_name = params['collection_name']

        self.dense_embeddings = HuggingFaceEmbeddings(model_name=dense_model,
                                                    model_kwargs=model_kwargs,
                                                    encode_kwargs=encode_kwargs)
        
        self.sparse_embeddings = FastEmbedSparse(model_name=sparse_model,
                                               parallel=20,
                                               batch_size=256)

        self.vector_store = QdrantVectorStore(client=self.client,
                                              collection_name=self.collection_name,
                                              embedding=self.dense_embeddings,
                                              sparse_embedding=self.sparse_embeddings,
                                              retrieval_mode=RetrievalMode.HYBRID,
                                              vector_name="dense",
                                              sparse_vector_name="sparse")

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})

        self.llm = ChatOllama(model='gemma4:31b', temperature=0.0) # type: ignore

        prompt_template = """You are a technical assistant. Answer the question based strictly on the provided context. If the answer is not in the context, say you do not know.

                             Context: 
                             {context}
                             
                             Question: {question}
                             Answer:"""


        self.prompt = ChatPromptTemplate.from_template(prompt_template)

        self.rerank = self._create_cross_encoder_reranker(model=CrossEncoder(params['rerank_model']), top_n=params['top_n'])

        self.join_docs = self._create_format_docs()

        self.rag_chain = (
                            {"docs": self.retriever,
                            "question": RunnablePassthrough()
                            }
                            | self.rerank
                            | self.join_docs
                            | self.prompt 
                            | self.llm 
                            | StrOutputParser()
                         )




    def _create_cross_encoder_reranker(self, model: CrossEncoder, top_n: int = 3) -> RunnableLambda:
        def rerank(inputs: dict) -> dict: # type: ignore
            query = inputs['question']
            docs = inputs['docs']

            if not docs:
                return {'docs': [], 'question': query}

            model_inputs = [[query, doc.page_content] for doc in docs]

            scores = model.predict(model_inputs)

            doc_score_pairs = list(zip(docs, scores))

            doc_score_pairs.sort(key=lambda x: x[1], reverse= True)

            return {'docs': [doc[0] for doc in doc_score_pairs[:top_n]], 'question': query}

        return RunnableLambda(rerank)

    def retrieve(self, query):
        return self.retriever.invoke(query)


    def _create_format_docs(self) -> RunnableLambda:
        def format_docs(data: dict) -> dict:
            data['context'] = "\n\n".join(doc.page_content for doc in data['docs'])
            return data
        return RunnableLambda(format_docs)


    def answer(self, query):
        print(self.rag_chain.invoke(query))