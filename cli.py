#!/usr/bin/env python3
"""MCP-LLM Command-Line Interface.

This module provides a command-line interface for interacting with
Large Language Models through the MCP (Model Control Protocol) interface.
"""

import sys
import typer
import asyncio
from loguru import logger
from typing import Optional, List, Any, Union
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

# Import client implementations
from mcp_llm.client.litellm import MCPClient as LiteLLMClient
from mcp_llm.client.anthropic import MCPClient as AnthropicClient
from mcp_llm.config import MCPConfig

# Initialize console
console = Console()

# Create Typer app
app = typer.Typer(
    name="mcp-llm",
    help="MCP Client CLI for interacting with Large Language Models and MCP servers",
)


def load_environment():
    """Load environment variables from multiple sources."""
    # Try standard .env file
    load_dotenv()

    # Try .env.local if it exists
    env_local = Path(".env.local")
    if env_local.exists():
        load_dotenv(env_local)

    # Try .env in /data if we're in a container environment
    data_env = Path("/data/.env")
    if data_env.exists():
        load_dotenv(data_env)

    # Try /data/.env.local if it exists
    data_env_local = Path("/data/.env.local")
    if data_env_local.exists():
        load_dotenv(data_env_local)


class ClientManager:
    """Manage LLM client lifecycle with proper error handling."""

    def __init__(
        self,
        client_type: str = "litellm",
        config: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "ollama/qwen2.5:32b",
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ):
        """Initialize client manager.

        Args:
            client_type: Type of client to use ("litellm" or "anthropic")
            config: Path to config file
            api_key: API key for the model provider
            model: Model to use
            max_tokens: Maximum tokens to generate
            base_url: Base URL for local or custom models
        """
        self.client_type = client_type
        self.config = config
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.client: Union[LiteLLMClient, AnthropicClient, Any] = None

        # Initialize client immediately
        if self.client_type == "litellm":
            self.client = LiteLLMClient(
                config_path=self.config,
                api_key=self.api_key,
                model=self.model,
                max_tokens=self.max_tokens,
                base_url=self.base_url,
            )
        elif self.client_type == "anthropic":
            self.client = AnthropicClient(
                config_path=self.config,
                anthropic_api_key=self.api_key,
                model=self.model,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unknown client type: {self.client_type}")

    async def connect_to_servers(self, server_names: Optional[List[str]] = None):
        """Connect to specified servers or all available servers."""
        if not self.client:
            raise ValueError("Client not initialized")

        if server_names:
            for server_name in server_names:
                try:
                    await self.client.connect_to_server(server_name)
                except Exception as e:
                    console.print(
                        f"[bold red]Error connecting to {server_name}:[/] {str(e)}"
                    )
        else:
            await self.client.connect_to_all_servers()

    async def process_query(self, query: str, system_prompt: str, temperature: float):
        """Process a query and yield the results."""
        if not self.client:
            raise ValueError("Client not initialized")

        async for chunk in self.client.process_query(
            query=query, system_prompt=system_prompt, temperature=temperature
        ):
            yield chunk

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.cleanup()


async def stream_to_console(console, query_generator):
    """Stream response chunks to the console with proper formatting.

    Args:
        console: Rich console instance to print to
        query_generator: Async generator producing response chunks
    """
    async for chunk in query_generator:
        console.print(chunk, end="", highlight=False)
    console.print()  # Add final newline


@app.command()
def chat(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to MCP configuration file"
    ),
    servers: Optional[List[str]] = typer.Option(
        None, "--server", "-s", help="Specific server(s) to connect to (default: all)"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key (default: use environment variable)"
    ),
    model: str = typer.Option(
        "anthropic/claude-3-7-sonnet-20241024", "--model", "-m", help="LLM model to use"
    ),
    max_tokens: int = typer.Option(
        4096, "--max-tokens", "-t", help="Maximum tokens for model completion"
    ),
    system_prompt: str = typer.Option(
        "You are a helpful assistant", "--system", help="System prompt for the LLM"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Temperature for model generation"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Single query to process (non-interactive mode)"
    ),
    client_type: str = typer.Option(
        "litellm", "--client", help="Client type: litellm or anthropic"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", "-b", help="Base URL for local models like Ollama"
    ),
):
    """Start an interactive chat with LLMs and MCP servers."""

    # Load environment variables
    load_environment()

    async def main():
        try:
            # Initialize client manager
            client_manager = ClientManager(
                client_type=client_type,
                config=config,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                base_url=base_url,
            )

            try:
                # Connect to servers
                await client_manager.connect_to_servers(servers)

                # Get available tools
                tools = client_manager.client.get_available_tools()
                if not tools:
                    console.print(
                        "[yellow]Warning:[/] No tools available from connected servers."
                    )
                else:
                    console.print(
                        f"[green]Connected to {len(client_manager.client.sessions)} servers with {len(tools)} tools available.[/]"
                    )

                # Single query mode
                if query:
                    console.print(f"[bold]Query:[/] {query}")
                    console.print("[bold]Response:[/]")

                    await stream_to_console(
                        console,
                        client_manager.process_query(
                            query=query,
                            system_prompt=system_prompt,
                            temperature=temperature,
                        ),
                    )

                    await client_manager.cleanup()
                    return

                # Interactive mode
                console.print("[bold green]MCP Client[/]")
                console.print(
                    "Type your queries below. Use [bold]exit[/], [bold]quit[/], or [bold]Ctrl+C[/] to exit."
                )
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
                        console.print("\n[bold purple]Assistant:[/] ", end="")
                        console.flush()
                        await stream_to_console(
                            console,
                            client_manager.process_query(
                                query=user_input,
                                system_prompt=system_prompt,
                                temperature=temperature,
                            ),
                        )

                    except KeyboardInterrupt:
                        console.print("\n[yellow]Interrupted by user[/]")
                        break
                    except Exception as e:
                        console.print(f"\n[bold red]Error:[/] {str(e)}")

                console.print("[bold green]Session ended[/]")

            finally:
                # Always ensure cleanup
                await client_manager.cleanup()

        except ValueError as e:
            console.print(f"[bold red]Error in configuration:[/] {str(e)}")
            sys.exit(1)
        except FileNotFoundError as e:
            console.print(f"[bold red]Error loading configuration:[/] {str(e)}")
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
    # Load environment variables
    load_environment()

    try:
        # Initialize config manager
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
    # Configure logging
    logger.remove()
    logger.add(
        RichHandler(rich_tracebacks=True, console=console),
        format="{message}",
        level="INFO",
    )

    # Run the app
    app()
