import argparse
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models

# inspect_chunks.py is a top-level script (run from src/), so push the repo root
# in so `from src.data import load_config` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import load_config

cfg = load_config()

PERSIST_DIR = cfg.persist_directory
COLLECTION_NAME = cfg.collection_name

# The Wikipedia appendix sections to filter out
EXCLUDED_SECTIONS = [
    'see also', 
    'references', 
    'further reading', 
    'external links', 
    'notes', 
    'bibliography'
]

def get_client() -> QdrantClient:
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(f"Vector store directory not found at {PERSIST_DIR}")
    return QdrantClient(path=PERSIST_DIR)


def list_available_topics(client: QdrantClient, limit: int = 2000):
    """Scans points to list all unique topics stored in the collection."""
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    topics = set()
    for r in records:
        topic = r.payload.get("metadata", {}).get("topic")
        if topic:
            topics.add(topic)

    print(f"\nAvailable Topics ({len(topics)} found):")
    print("-" * 50)
    for topic in sorted(topics):
        print(f"  • {topic}")
    print("-" * 50)


def fetch_chunks_by_topic(client: QdrantClient, topic_name: str):
    """Retrieves and prints all chunks matching a specific topic, filtering out appendices."""
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.topic",
                    match=models.MatchValue(value=topic_name),
                )
            ]
        ),
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    # Fallback search if exact match doesn't hit
    if not records:
        all_records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=2000,
            with_payload=True,
            with_vectors=False,
        )
        records = [
            r for r in all_records
            if topic_name.lower() in str(r.payload.get("metadata", {}).get("topic", "")).lower()
        ]

    if not records:
        print(f"\nNo chunks found matching topic: '{topic_name}'")
        return

    filtered_records = []
    dropped_count = 0

    # Filter out the unwanted sections
    for r in records:
        headings = r.payload.get("metadata", {}).get("headings", "").lower()
        # If any excluded term is in the heading, skip this chunk
        if any(excluded in headings for excluded in EXCLUDED_SECTIONS):
            dropped_count += 1
            continue
        filtered_records.append(r)

    print(f"\nFound {len(filtered_records)} clean chunks for topic: '{topic_name}'")
    if dropped_count > 0:
        print(f"(Filtered out {dropped_count} chunks belonging to References/External Links)")
    print("=" * 80)

    for idx, r in enumerate(filtered_records, 1):
        metadata = r.payload.get("metadata", {})
        headings = metadata.get("headings", "Top-level Section")
        content = r.payload.get("page_content", "")
        doc_id = r.id

        print(f"\n[Chunk #{idx}] (ID: {doc_id})")
        print(f"Section : {headings}")
        print(f"Length  : {len(content.split())} words")
        print("-" * 80)
        print(content.strip())
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect chunks stored in Qdrant by topic.")
    parser.add_argument("--list", action="store_true", help="List all unique topics available in Qdrant.")
    parser.add_argument("--topic", type=str, help="The specific topic to retrieve chunks for.")

    args = parser.parse_args()
    client = get_client()

    if args.list or not args.topic:
        list_available_topics(client)
        if not args.topic:
            print("\nRun with '--topic \"<Topic Name>\"' to view individual chunks for that topic.")
    else:
        fetch_chunks_by_topic(client, args.topic)

    client.close()