# Example usage
import anyio
from mcp_llm.client.litellm import MCPClient
from typing import AsyncGenerator
from contextlib import aclosing


async def main():
    client = MCPClient(
        model="ollama_chat/qwen2.5:32b",  # Just the model name without provider prefix
        base_url="http://10.13.1.11:11434"
    )
    async with anyio.create_task_group() as tg:
        try:
            # Connect to servers
            await client.connect_to_all_servers()
            # Process query
            query = "what's in /data directory?"
            print(f"running: {query}")
            async with aclosing(client.process_query(query)) as query_gen:
                async for chunk in query_gen:
                    print(chunk, end="", flush=True)
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            # Make sure to clean up any resources
            try: 
                if hasattr(client, 'close') and callable(client.close):
                    await client.close()
                elif hasattr(client, 'aclose') and callable(client.aclose):
                    await client.aclose()
            except Exception as ex:
                print(f"ERR: {ex}")

if __name__ == "__main__":
    try:
        anyio.run(main)
    except RuntimeError as ex:
        print(f"Error: {ex}")

