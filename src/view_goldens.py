import json
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

def view_dataset(filepath: str):
    console = Console()
    
    if not os.path.exists(filepath):
        console.print(f"[bold red]File not found: {filepath}[/bold red]")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both DeepEval dict format {"goldens": [...]} and raw list format [...]
    goldens = data.get("goldens", data) if isinstance(data, dict) else data

    console.print(f"\n[bold green]✅ Loaded {len(goldens)} test cases from {filepath}[/bold green]\n")

    for idx, item in enumerate(goldens):
        question = item.get("input", "N/A")
        expected = item.get("expected_output", "N/A")
        
        raw_context = item.get("context", ["N/A"])
        context_str = "\n".join(raw_context) if isinstance(raw_context, list) else str(raw_context)
        
        # Truncate context if it's massively long to keep the terminal readable
        if len(context_str) > 1200:
            context_str = context_str[:1200] + "\n\n... [CONTEXT TRUNCATED FOR DISPLAY]"

        console.print(f"[bold cyan]━━━ Test Case {idx + 1} of {len(goldens)} ━━━[/bold cyan]")
        
        console.print(Panel(
            question, 
            title="[bold yellow]❓ Question[/bold yellow]", 
            border_style="yellow"
        ))
        
        console.print(Panel(
            expected, 
            title="[bold green]🎯 Expected Answer (Golden)[/bold green]", 
            border_style="green"
        ))
        
        console.print(Panel(
            context_str, 
            title="[bold blue]📄 Provided Context Chunk[/bold blue]", 
            border_style="blue"
        ))

        # Navigation prompt
        nav = Prompt.ask("[bold magenta]Press Enter for next, type 'q' to quit[/bold magenta]")
        if nav.lower() == 'q':
            console.print("[bold red]Exiting viewer.[/bold red]")
            break
        console.clear()

if __name__ == "__main__":
    # Point this to whichever file you want to inspect
    # target_file = "data/eval/goldens_clean.json"
    target_file = 'data/eval/manual_eval.json'
    
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    view_dataset(target_file)