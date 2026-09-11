# 2_chunk_data.py
import io
import json
import logging
from rich.console import Console
from rich.progress import track

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.chunking import HierarchicalChunker

logging.getLogger("docling").setLevel(logging.ERROR)
console = Console()

def chunk_dataset(input_filepath: str, output_filepath: str):
    console.print(f"[bold cyan]Loading raw HTML from {input_filepath}...[/bold cyan]")
    
    # 1. Load the raw dataset
    with open(input_filepath, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    converter = DocumentConverter()
    chunker = HierarchicalChunker()
    all_ready_chunks = []

    # 2. Process each document through Docling
    for doc in track(raw_dataset, description="Chunking documents via Docling..."):
        page_title = doc["metadata"]["topic"]
        raw_html = doc["raw_html"]
        
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
                chunk_meta = doc["metadata"].copy() # Get the global metadata from Script 1
                chunk_meta["chunk_id"] = i
                
                if chunk.meta.headings:
                     chunk_meta["headings"] = " > ".join(chunk.meta.headings)
                else:
                     chunk_meta["headings"] = "Intro/No Heading"

                all_ready_chunks.append({
                    "page_content": chunk.text,
                    "metadata": chunk_meta
                })
        except Exception as e:
            console.print(f"\n[bold red]Docling failed on {page_title}: {e}[/bold red]")

    # 4. Save the fully chunked dataset
    console.print(f"\n[bold yellow]Saving chunked dataset to {output_filepath}...[/bold yellow]")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(all_ready_chunks, f, indent=4, ensure_ascii=False)
        
    console.print(f"[bold green]Success! {len(all_ready_chunks)} total chunks saved.[/bold green]")

if __name__ == "__main__":
    chunk_dataset(
        input_filepath="data/raw/ai_ml_raw_html.json",
        output_filepath="data/processed/ai_ml_rag_chunks.json"
    )