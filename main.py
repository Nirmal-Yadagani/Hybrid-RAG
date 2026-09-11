import sys
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()

def main():
    console = Console()
    console.print("[bold blue]======================================[/bold blue]")
    console.print("[bold green]🤖 Hybrid-RAG Wikipedia Assistant Initialized[/bold green]")
    console.print("[bold blue]======================================[/bold blue]")
    console.print("Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.\n")
    
    # 1. Initialize the bot (Load Qdrant, CrossEncoder, and Ollama)
    with console.status("[bold yellow]Booting up Embeddings, Qdrant, and LLM...[/bold yellow]", spinner="dots"):
        from src.retriever import QAbot
        bot = QAbot()
    
    # 2. Continuous Conversation Loop
    while True:
        try:
            # Get user input with a nice cyan prompt
            query = console.input("[bold cyan]Query:[/bold cyan] ").strip()
            
            # Exit conditions
            if query.lower() in ['exit', 'quit']:
                console.print("[bold red]Shutting down connections... Goodbye![/bold red]")
                break
            if not query:
                continue

            # 3. Retrieve and Generate with a loading spinner
            with console.status("[bold yellow]Retrieving & Generating...[/bold yellow]", spinner="bouncingBar"):
                response = bot.answer(query)
            
            # 4. Print the final answer rendered as beautiful Markdown!
            console.print("\n[bold magenta]Answer:[/bold magenta]")
            console.print(Markdown(response))
            console.print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            # Handles Ctrl+C gracefully
            console.print("\n[bold red]Force quitting... Goodbye![/bold red]")
            sys.exit(0)

if __name__ == "__main__":
    main()