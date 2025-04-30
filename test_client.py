#!/usr/bin/env python3
"""Test client for MCP-LLM integration."""

import sys
import anyio
import logging
from typing import AsyncGenerator
from contextlib import aclosing, AsyncExitStack

from loguru import logger
from mcp_llm.client.litellm import MCPClient

MODEL = "ollama_chat/qwen3:32b-fp16"
OLLAMA_BASE_URL = "http://10.13.1.11:11434"

async def main() -> None:
    """Run a test query against the MCP client with proper error handling."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Create an exit stack to ensure proper cleanup
    async with AsyncExitStack() as stack:
        try:
            # Initialize client with proper error handling
            client = MCPClient(
                model=MODEL,
                base_url=OLLAMA_BASE_URL
            )
            
            # Register the client's cleanup method with our exit stack
            stack.push_async_callback(client.cleanup)
            
            # Connect to servers
            logger.info("Connecting to MCP servers...")
            await client.connect_to_all_servers()
            
            # Get available tools
            tools = client.get_available_tools()
            logger.info(f"Connected to {len(client.sessions)} servers with {len(tools)} tools available")
            
            # Process query
            query = "what's in /data directory?"
            logger.info(f"Running query: {query}")
            
            # Properly handle streaming response
            async with aclosing(client.process_query(query)) as query_gen:
                try:
                    async for chunk in query_gen:
                        print(chunk, end="", flush=True)
                    print()  # Add newline at the end
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    raise
            
        except FileNotFoundError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            sys.exit(2)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            sys.exit(3)

if __name__ == "__main__":
    try:
        anyio.run(main)
    except KeyboardInterrupt:
        logger.info("Test client interrupted by user")
        sys.exit(0)
    except RuntimeError as ex:
        logger.error(f"Runtime error: {ex}")
        sys.exit(4)
