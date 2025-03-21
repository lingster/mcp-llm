import sys
import typer
import asyncio
from loguru import logger
from typing import Optional, List

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from mcp_llm import MCPClient

console = Console()

# Load environment variables from .env file
load_dotenv()

# Create Typer app
app = typer.Typer(
    name="mcp-llm",
    help="MCP Client CLI for interacting with Anthropic's Claude model and MCP servers"
)

@app.command()
def chat(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to MCP configuration file"
    ),
    servers: Optional[List[str]] = typer.Option(
        None, "--server", "-s", help="Specific server(s) to connect to (default: all)"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="Anthropic API key (default: ANTHROPIC_API_KEY env var)"
    ),
    model: str = typer.Option(
        "claude-3-7-sonnet-latest", "--model", "-m", help="Claude model to use"
    ),
    max_tokens: int = typer.Option(
        4096, "--max-tokens", "-t", help="Maximum tokens for model completion"
    ),
    system_prompt: Optional[str] = typer.Option(
        "You are a helpful assistant", "--system", help="System prompt for Claude"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Temperature for model generation"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Single query to process (non-interactive mode)"
    ),
):
    """Start an interactive chat with Claude and MCP servers."""

    async def main():
        try:
            # Initialize client
            client = MCPClient(
                config_path=config,
                anthropic_api_key=api_key,
                model=model,
                max_tokens=max_tokens
            )

            # Connect to servers
            if servers:
                for server_name in servers:
                    try:
                        await client.connect_to_server(server_name)
                    except Exception as e:
                        console.print(f"[bold red]Error connecting to {server_name}:[/] {str(e)}")
            else:
                await client.connect_to_all_servers()

            # Get available tools
            tools = client.get_available_tools()
            if not tools:
                console.print("[yellow]Warning:[/] No tools available from connected servers.")
            else:
                console.print(f"[green]Connected to {len(client.sessions)} servers with {len(tools)} tools available.[/]")

            # Single query mode
            if query:
                console.print(f"[bold]Query:[/] {query}")
                console.print("[bold]Response:[/]")

                # Process query with streaming
                async for chunk in await client.process_query(
                    query=query,
                    system_prompt=system_prompt,
                    temperature=temperature
                ):
                    print(chunk, end="", flush=True)
                print()  # Add final newline

                await client.cleanup()
                return

            # Interactive mode
            console.print("[bold green]MCP Client[/]")
            console.print("Type your queries below. Use [bold]exit[/], [bold]quit[/], or [bold]Ctrl+C[/] to exit.")
            console.print("=========================================")

            while True:
                try:
                    # Get user input
                    console.print("\n[bold blue]You:[/]", end=" ")
                    user_input = input()

                    # Check for exit commands
                    if user_input.lower() in ["exit", "quit", "q"]:
                        break

                    # Display assistant response
                    console.print("[bold purple]Claude:[/]", end=" ")

                    # Process query with streaming
                    async for chunk in await client.process_query(
                        query=user_input,
                        system_prompt=system_prompt,
                        temperature=temperature
                    ):
                        print(chunk, end="", flush=True)
                    print()  # Add final newline

                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted by user[/]")
                    break
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/] {str(e)}")

            await client.cleanup()
            console.print("[bold green]Session ended[/]")
        except ValueError as e:
            console.print(f"[bold red]Error listing servers:[/] {str(e)}")
            sys.exit(1)

        except Exception as e:
            console.print(f"[bold red]Critical error:[/] {str(e)}")
            logger.exception(f"Critical error:{e}")
            sys.exit(1)

    # Run async code
    asyncio.run(main())

@app.command()
def servers(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to MCP configuration file"
    ),
):
    """List configured MCP servers."""
    try:
        from mcp_llm.config import MCPConfig
        config_manager = MCPConfig(config)
        config_manager.load_config()

        server_names = config_manager.list_servers()

        if not server_names:
            console.print("[yellow]No MCP servers configured.[/]")
            return

        console.print(f"[bold green]Configured MCP Servers ({len(server_names)}):[/]")
        for name in server_names:
            console.print(f"- {name}")

        console.print(f"\nConfig loaded from: [italic]{config_manager.config_path}[/]")
    except Exception as e:
        console.print(f"[bold red]Error listing servers:[/] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    app()
