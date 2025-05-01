import asyncio
from mcp_llm.client.litellm import MCPClient
from loguru import logger

MODEL = "openai/qwen3:32b"
OLLAMA_BASE_URL = "http://10.13.1.11:30000/v1"
API_KEY = "sk-..."

async def test_mcp_client():
    """Test the MCPClient with a simple query."""
    # You may need to set your API key
    # os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
    # Create client
    client = MCPClient(
        # Use default config path or set a specific one
        # config_path="path/to/config.json"
        model=MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key=API_KEY,
        debug=False,
    )
    
    # Connect to all servers
    await client.connect_to_all_servers()
    tools = client.get_available_tools()
    logger.info(f"Connected to {len(client.sessions)} servers with {len(tools)} tools available")

    try:
        # Async version
        async def do_async():
            print("Testing async version:")
            response_text = ""
            # Properly consume the async generator by awaiting each chunk
            async for chunk in client.aprocess_query(
                "What is the current time and what tools are available?",
                system_prompt="You are a helpful AI assistant with access to tools."
            ):
                print(chunk, end="", flush=True)
                response_text += chunk
            
            # Return the complete response for debugging
            print("\n\n" + "-"*50 + "\n\n")
            return response_text
        
        # Sync version
        def do_sync():
            print("Testing sync version:")
            response = client.process_query(
                "What is in the /data folder? /no_think",
                system_prompt="You are a helpful AI assistant with access to tools."
            )
            print(response)
            return response

        # Run the async version
        response_async = await do_async()
        print(f"Async response length: {len(response_async) if response_async else 0}")
        
        # Uncomment to run the sync version
        #response_sync = do_sync()
        #print(f"Sync response length: {len(response_sync) if response_sync else 0}")

    except Exception as ex:
        logger.error(f"Error: {ex}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(test_mcp_client())
