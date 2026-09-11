import os
import random

from deepeval.dataset import EvaluationDataset
from deepeval.models import OllamaModel
from deepeval.synthesizer import Synthesizer

from retriever import QAbot

rag_bot = QAbot()

# 2. Fetch a larger pool of seed chunks to ensure we can sample 50 unique points
records, _ = rag_bot.client.scroll(
    rag_bot.collection_name, 
    limit=500, 
    with_payload=True, 
    with_vectors=False
)

# Sample 50 documents (50 contexts * 2 goldens per context = 100 goldens)
seed_points = random.sample(records, 50)

contexts = []
payload_text_key = "page_content"

print("Retrieving and reranking contexts via CrossEncoder...")
for point in seed_points:
    # Optional: Truncate the seed text if your retriever prefers shorter queries 
    # over full 1000-token document embeddings
    seed_text = point.payload.get(payload_text_key)
    if not seed_text:
        continue

    retrieved_docs = rag_bot.retrieve(seed_text)
    
    # Inject our newly scraped Wikipedia metadata into the Synthesizer's context!
    context_group = []
    for doc in retrieved_docs:
        # Assuming your QAbot returns LangChain Document objects with .metadata
        topic = doc.metadata.get("topic", "Unknown Topic")
        heading = doc.metadata.get("headings", "Unknown Section")
        
        # Format a rich string for the LLM to read
        rich_context = f"DOCUMENT TOPIC: {topic}\nSECTION: {heading}\nCONTENT:\n{doc.page_content}"
        context_group.append(rich_context)
        
    contexts.append(context_group)

ollama_llm = OllamaModel(
    model='qwen3.6:latest', 
    base_url="http://localhost:11434",
    temperature=0.7 
)

synthesizer = Synthesizer(model=ollama_llm)

# 3. Process in chunks to prevent Ollama queue timeouts
all_goldens = []
chunk_size = 10  # The A6000 handles chunks of 10 effortlessly

print(f"\nStarting generation of 100 goldens in chunks of {chunk_size} contexts...")
for i in range(0, len(contexts), chunk_size):
    chunk_num = (i // chunk_size) + 1
    total_chunks = (len(contexts) + chunk_size - 1) // chunk_size
    print(f"Processing chunk {chunk_num} of {total_chunks}...")
    
    context_chunk = contexts[i : i + chunk_size]
    
    chunk_goldens = synthesizer.generate_goldens_from_contexts(
        contexts=context_chunk,
        include_expected_output=True,
        max_goldens_per_context=2
    )
    all_goldens.extend(chunk_goldens)

# 4. Save the combined dataset
eval_dataset = EvaluationDataset(goldens=all_goldens)

os.makedirs('data/eval', exist_ok=True)
# Saved under a new name to avoid overwriting your previous test run
eval_dataset.save_as(file_name='goldens_100', file_type='json', directory='data/eval')

rag_bot.client.close()
print("\n✅ Synthetic evaluation dataset with 100 goldens successfully generated!")