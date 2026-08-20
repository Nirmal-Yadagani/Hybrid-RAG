import random

from deepeval.dataset import EvaluationDataset
from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer

from retriever import QAbot

rag_bot = QAbot()

records, _ = rag_bot.client.scroll(rag_bot.collection_name, limit=100, with_payload=True, with_vectors=False)

seed_points = random.sample(records, 5)

contexts = []
payload_text_key = "page_content"

for point in seed_points:
    seed_text = point.payload.get(payload_text_key) # type: ignore
    if not seed_text:
        continue

    retrieved_docs = rag_bot.retriever.invoke(seed_text)

    context_group = [seed_text] + [doc.page_content for doc in retrieved_docs]

    contexts.append(context_group)

        
ollama_llm = OllamaModel(model='qwen3.6:latest',
                        base_url="http://localhost:11434",
                        temperature=0)

synthesizer = Synthesizer(model=ollama_llm)


goldens = synthesizer.generate_goldens_from_contexts(contexts=contexts,
                                                     include_expected_output=True,
                                                     max_goldens_per_context=2)


eval_dataset = EvaluationDataset(goldens=goldens)

eval_dataset.save_as(file_name='goldens', file_type='json', directory='data/eval')