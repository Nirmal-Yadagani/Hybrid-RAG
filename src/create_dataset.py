import json
import time
import logging
import os
import sys
from datetime import datetime
from rich.console import Console
from rich.progress import track
from scrapling import StealthyFetcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import StageLogger  # noqa: E402

logging.getLogger("scrapling").setLevel(logging.WARNING)
console = Console()
log = StageLogger("dataset")

def extract_wikipedia_page(topic_url: str) -> dict:
    """Fetches clean HTML and global metadata without chunking."""
    response = StealthyFetcher.fetch(topic_url, headless=True)
    if response.status != 200:
         raise Exception(f"Failed to fetch {topic_url}: Status {response.status}")

    page = response

    # 1. Global Metadata
    title_elements = page.css('title')
    page_title = str(title_elements[0].text) if (title_elements and title_elements[0].text) else str(topic_url.split('/')[-1])

    global_metadata = {
        "source_url": str(topic_url),
        "topic": page_title,
        "source_type": "Wikipedia",
        "extraction_date": datetime.now().strftime("%Y-%m-%d"),
    }

    # 2. Precision HTML Extraction
    body_elements = page.css('body')
    main_content_html = str(body_elements[0].html_content) if body_elements else str(response.text)

    # Return the un-chunked payload
    return {
        "metadata": global_metadata,
        "raw_html": main_content_html # Saving HTML preserves the structure for Docling later!
    }

def build_raw_dataset(url_list: list[str], output_filepath: str):
    master_dataset = []
    failed_urls = []
    started = time.perf_counter()

    console.print(f"[bold cyan]Starting RAW HTML extraction of {len(url_list)} topics...[/bold cyan]")
    log.info("dataset.start", "Starting raw HTML extraction", url_count=len(url_list))

    for url in track(url_list, description="Downloading pages..."):
        try:
            payload = extract_wikipedia_page(url)
            master_dataset.append(payload)
            time.sleep(1.5)
        except Exception as e:
            console.print(f"\n[bold red]Error processing {url}: {e}[/bold red]")
            failed_urls.append(url)
            log.warning("dataset.url_failed", "URL extraction failed", url=url, error=str(e))
            time.sleep(2)

    console.print(f"\n[bold yellow]Saving raw dataset to {output_filepath}...[/bold yellow]")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(master_dataset, f, indent=4, ensure_ascii=False)

    elapsed = time.perf_counter() - started
    console.print(f"[bold green]Saved {len(master_dataset)} raw documents![/bold green]")
    log.info(
        "dataset.done",
        "Raw HTML extraction completed",
        saved=len(master_dataset),
        failed=len(failed_urls),
        elapsed_s=round(elapsed, 2),
    )

if __name__ == "__main__":

    import yaml

    with open('params.yaml', 'r') as config:
        params = yaml.safe_load(config)

    with open(params['urls_path'], 'r') as f:
        ai_ml_wikipedia_urls = [url.strip() for url in f.readlines() if not url.startswith('#')]
    
    build_raw_dataset(ai_ml_wikipedia_urls, "data/raw/ai_ml_raw_html.json")