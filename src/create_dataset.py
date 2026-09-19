import json
import time
import logging
import os
import sys
from datetime import datetime
from rich.console import Console
from rich.progress import track
from scrapling import StealthyFetcher
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import StageLogger  # noqa: E402

logging.getLogger("scrapling").setLevel(logging.WARNING)
console = Console()
log = StageLogger("dataset")

def extract_wikipedia_page(topic_url: str) -> dict:
    """Fetches clean HTML from the Wikipedia API and surgically removes reference sections."""
    response = StealthyFetcher.fetch(topic_url, headless=True)
    if response.status != 200:
         raise Exception(f"Failed to fetch {topic_url}: Status {response.status}")

    page = response

    # 1. Global Metadata
    title_elements = page.css('title')
    page_title = str(title_elements[0].text) if (title_elements and title_elements[0].text) else str(topic_url.split('/')[-1])
    page_title = page_title.replace(" - Wikipedia", "").strip()

    global_metadata = {
        "source_url": str(topic_url),
        "topic": page_title,
        "source_type": "Wikipedia",
        "extraction_date": datetime.now().strftime("%Y-%m-%d"),
    }

    # 2. Extract the Body (Compatible with the REST API)
    body_elements = page.css('body')
    raw_html = str(body_elements[0].html_content) if body_elements else str(response.text)

    # 3. Aggressive Noise Filtering
    soup = BeautifulSoup(raw_html, 'html.parser')

    # A. Delete known Wikipedia UI/Noise classes and non-text elements
    noise_selectors = [
        'style',             # Kills all inline CSS
        'script',            # Kills all javascript
        'table.sidebar',     # Kills the "Part of a series on..." right-hand navigation boxes
        'div.hatnote',       # Kills "Main article:" and "See also:" disclaimers
        '.reflist',          # Reference lists
        '.navbox',           # Bottom navigation boxes
        '.vertical-navbox',  # Side navigation boxes
        '.metadata',         # Page warning banners
        '.printfooter',      # Print URLs
        '.mw-editsection',   # "[edit]" links next to headings
        'sup.reference',     # Inline citation numbers like [1], [2]
        'sup.noprint',       # "citation needed" brackets
        'div.shortdescription', # Hidden metadata descriptions
        'figure',            # Kills image captions and thumbnails (optional, but good for pure text RAG)
        '#toc',              # Kills the Table of Contents box
        '.toc',              # Kills alternative TOC classes
        '.thumb',            # Kills image thumbnails and their captions
        'video',             # Kills video embed tags
        '.mw-empty-elt'      # Kills empty spans used for MediaWiki tracking
    ]
    
    for selector in noise_selectors:
        for element in soup.select(selector):
            element.decompose()

    # B. Delete the Appendices (See also, References, Further reading, External links)
    appendix_ids = ['See_also', 'References', 'Further_reading', 'External_links', 'Notes', 'Bibliography']
    for header_id in appendix_ids:
        header = soup.find(id=header_id)
        if header:
            # 🚨 FIX: Find the parent <section> instead of the <h2>
            parent_section = header.find_parent('section')
            if parent_section:
                # Delete all <section> blocks that come AFTER this one
                for sibling in parent_section.find_next_siblings():
                    sibling.decompose()
                # Delete the appendix section itself
                parent_section.decompose()

    return {
        "metadata": global_metadata,
        "raw_html": str(soup)
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
        ai_ml_wikipedia_urls = []
        for line in f:
            # Strip whitespace, newlines, and stray quotes
            clean_url = line.strip().strip('"').strip("'")
            # Ignore empty lines and comments
            if clean_url and not clean_url.startswith('#'):
                ai_ml_wikipedia_urls.append(clean_url)
    
    build_raw_dataset(ai_ml_wikipedia_urls, "data/raw/ai_ml_raw_html.json")