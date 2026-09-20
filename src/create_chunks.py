import sys
import time
import os

# Stages run via DVC as `python src/<stage>.py`, which puts only ``src/`` on the
# path. Push the repo root (``src/data``'s parent) in so ``from src...`` imports
# resolve from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import io
import logging
from rich.console import Console
from rich.progress import track
from dotenv import load_dotenv

from src.logger import StageLogger
from src.data import (
    ChunkRecord,
    RawPage,
    load_config,
    load_json,
    write_json_array,
)

load_dotenv()

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.chunking import HybridChunker

logging.getLogger("docling").setLevel(logging.ERROR)
console = Console()
log = StageLogger("chunk")

def chunk_dataset(input_filepath: str, output_filepath: str):
    console.print(f"[bold cyan]Loading raw HTML from {input_filepath}...[/bold cyan]")

    cfg = load_config()

    raw_dataset: list[RawPage] = load_json(input_filepath, model=RawPage)

    converter = DocumentConverter()
    chunker = HybridChunker(tokenizer=cfg.dense_embedding_model,  # Bounds chunks to model tokenizer limits
                            max_tokens=cfg.chunk_max_tokens,      # Caps maximum chunk size at ~250-300 words
                            merge_peers=cfg.chunk_merge_peers)    # Automatically merges consecutive micro-chunks
    log.info(
        "chunk.start",
        "Chunking started",
        tokenizer=cfg.dense_embedding_model,
        max_tokens=cfg.chunk_max_tokens,
        merge_peers=cfg.chunk_merge_peers,
        docs=len(raw_dataset),
    )

    all_ready_chunks = []
    started = time.perf_counter()

    # 2. Process each document through Docling
    for doc in track(raw_dataset, description="Chunking documents via Docling..."):
        page_title = doc.metadata.topic
        raw_html = doc.raw_html

        # Pass the sterilized HTML to Docling
        clean_html_bytes = raw_html.encode('utf-8')
        stream = DocumentStream(name=page_title, stream=io.BytesIO(clean_html_bytes))
        
        try:
            conversion_result = converter.convert(stream)
            docling_document = conversion_result.document
            
            # CHUNK the document using Docling
            chunks = chunker.chunk(docling_document)
            
            # 3. Format chunks and inject metadata
            for i, chunk in enumerate(chunks):
                chunk_meta = doc.metadata.model_copy() # Get the global metadata from Script 1
                chunk_meta.chunk_id = i
                
                if chunk.meta.headings:
                     chunk_meta.headings = " > ".join(chunk.meta.headings)
                else:
                     chunk_meta.headings = "Intro/No Heading"

                all_ready_chunks.append(
                    ChunkRecord.model_validate({
                        "page_content": chunk.text,
                        "metadata": chunk_meta.model_dump(),
                    })
                )

        except Exception as e:
            console.print(f"\n[bold red]Docling failed on {page_title}: {e}[/bold red]")

    # Discard pure noise fragments before saving
    cleaned_chunks = [
        c for c in all_ready_chunks
        if len(c.page_content.split()) >= 20
    ]

    # 4. Save the fully chunked dataset
    console.print(f"\n[bold yellow]Saving chunked dataset to {output_filepath}...[/bold yellow]")
    write_json_array(output_filepath, cleaned_chunks)

    console.print(f"[bold green]Success! {len(all_ready_chunks)} total chunks saved.[/bold green]")
    log.info(
        "chunk.done",
        "Chunking completed",
        raw_chunks=len(all_ready_chunks),
        saved_chunks=len(cleaned_chunks),
        dropped=len(all_ready_chunks) - len(cleaned_chunks),
        elapsed_s=round(time.perf_counter() - started, 2),
    )

if __name__ == "__main__":
    chunk_dataset(
        input_filepath="data/raw/ai_ml_raw_html.json",
        output_filepath="data/processed/ai_ml_rag_chunks.json"
    )