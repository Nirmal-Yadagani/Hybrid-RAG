import os
import atexit
import yaml
import weave
# Initialize Weave tracing for Langchain
weave_client = weave.init("hybrid-rag-traces")
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
        with open('params.yaml', 'r') as config:
            params = yaml.safe_load(config)

        dense_model = params['dense_embedding_model']
        sparse_model = params['sparse_embedding_model']
        hf_token = os.getenv('HF_TOKEN')
        model_kwargs = {'device': 'cuda', 'token': hf_token}
        encode_kwargs = {'normalize_embeddings': params['normalize_embeddings']}
        
        self.client = QdrantClient(path=params['persist_directory'])
        self.connections = [self.client]
        atexit.register(self.close_connections)
        self.collection_name = params['collection_name']

        # Key the weave cost table to whichever chat model is configured
        weave_client.add_cost(
            llm_id=params['chat_model'],
            prompt_token_cost=0.14 / 1_000_000,
            completion_token_cost=0.40 / 1_000_000
        )

        self.dense_embeddings = HuggingFaceEmbeddings(
            model_name=dense_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        self.sparse_embeddings = FastEmbedSparse(
            model_name=sparse_model,
            parallel=20,
            batch_size=256
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse"
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": params['retrieval_k']})
        self.llm = ChatOllama(model=params['chat_model'], temperature=params['chat_temperature'])

        prompt_template = """You are a precise technical assistant. Answer the question based ONLY on the provided documents. 
        If the answer is not in the context, say "I do not know".
        
        CRITICAL INSTRUCTION: Every factual claim in your answer MUST include an inline citation to the document ID it came from. 
        Format your citations exactly like this: [Doc 1] or [Doc 1, Doc 2].

        Context: 
        {context}
        
        Question: {question}
        Answer:"""

        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.rerank = self._create_cross_encoder_reranker(
            model=CrossEncoder(params['rerank_model']),
            top_n=params['top_n'],
            score_threshold=params['rerank_score_threshold']
        )
        self.join_docs = self._create_format_docs()

        # --- ARCHITECTURE FIX: Modular Chains ---
        
        # 1. The Retrieval Sub-Chain (Retrieves Top 20 -> Reranks to Top N)
        self.retrieval_chain = (
            {"docs": self.retriever, "question": RunnablePassthrough()}
            | self.rerank
        )

        # 2. The Generation Sub-Chain (Formats -> Prompts -> LLM)
        self.generation_chain = (
            self.join_docs 
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )

        # 3. The Full RAG Chain (Combines 1 and 2 for standard runtime)
        self.rag_chain = self.retrieval_chain | self.generation_chain

    def _create_cross_encoder_reranker(self, model: CrossEncoder, top_n: int = 3, score_threshold: float = 0.0) -> RunnableLambda:
        def rerank(inputs: dict) -> dict:
            query = inputs['question']
            docs = inputs['docs']

            if not docs:
                return {'docs': [], 'question': query}

            model_inputs = [[query, doc.page_content] for doc in docs]
            scores = model.predict(model_inputs)
            doc_score_pairs = [(doc, score) for doc, score in zip(docs, scores) if score > score_threshold]
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            filtered_docs = [doc for doc, score in doc_score_pairs[:top_n]]

            if not filtered_docs and docs:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                filtered_docs = [docs[best_idx]]

            return {'docs': filtered_docs, 'question': query}
        return RunnableLambda(rerank)

    def _create_format_docs(self) -> RunnableLambda:
        def format_docs(data: dict) -> dict:
            formatted_contexts = []
            
            for i, doc in enumerate(data['docs']):
                # Extract the metadata we injected during Docling extraction
                topic = doc.metadata.get("topic", "Unknown Topic")
                headings = doc.metadata.get("headings", "No Section")
                
                # Create a rich, contextual block for the LLM
                rich_chunk = (
                    f"<document id='{i+1}'>\n"
                    f"SOURCE: {topic}\n"
                    f"HEADINGS: {headings}\n"
                    f"TEXT: {doc.page_content}\n"
                    f"</document>"
                )
                formatted_contexts.append(rich_chunk)
                
            # Join the rich blocks together with double newlines
            data['context'] = "\n\n".join(formatted_contexts)
            return data
            
        return RunnableLambda(format_docs)

    def retrieve(self, query: str):
        """Returns ONLY the reranked documents."""
        result = self.retrieval_chain.invoke(query)
        return result['docs']

    @weave.op()
    def answer(self, query: str):
        """Standard runtime method: returns just the string answer."""
        return self.rag_chain.invoke(query)

    @weave.op()
    def ask_and_retrieve(self, query: str):
        """
        EVALUATION METHOD: Runs the CrossEncoder once, passing the exact same 
        reranked docs to the LLM and back to you.
        """
        # Step 1: Get reranked docs
        retrieval_result = self.retrieval_chain.invoke(query)
        
        # Step 2: Pass those exact docs to the LLM
        answer = self.generation_chain.invoke(retrieval_result)
        
        return {
            "answer": answer,
            "docs": retrieval_result['docs']
        }

    def close_connections(self):
        for conn in self.connections:
            try:
                conn.close()
            except Exception:
                pass