from enum import Enum
import os
import sys
from loguru import logger
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (RawMessageStartEvent, RawContentBlockStartEvent, RawMessageStopEvent, RawMessageDeltaEvent,
                             RawContentBlockStopEvent, RawContentBlockDeltaEvent)
from anthropic.types import MessageParam, ToolUseBlock, TextBlock
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Resource, Tool

from .config import MCPConfig

logger.remove()
logger.add(sys.stderr, level="INFO")


class MCPClient:
    """MCP client for interacting with Claude and MCP servers."""

    # Use a separator that won't appear in server or tool names
    TOOL_NAME_SEPARATOR = "__"

    def __init__(
        self,
        config_path: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        model: str = "claude-3-7-sonnet-latest",
        max_tokens: int = 4096
    ):
        """Initialize the MCP client.

        Args:
            config_path: Path to the MCP configuration file.
            anthropic_api_key: Anthropic API key, defaults to ANTHROPIC_API_KEY env var.
            model: Claude model to use.
            max_tokens: Maximum tokens for model completion.
        """
        self.config = MCPConfig(config_path)
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY env var or pass it to the constructor.")

        self.model = model
        self.max_tokens = max_tokens
        self.anthropic = Anthropic(api_key=self.api_key)
        self.async_anthropic = AsyncAnthropic(api_key=self.api_key)

        # MCP server connections and state
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.server_tools: Dict[str, List[Tool]] = {}
        
        # Map fully-qualified tool names to (server_name, tool_name) tuples
        self.tool_map: Dict[str, Tuple[str, str]] = {}

    async def connect_to_server(self, server_name: str) -> None:
        """Connect to a specific MCP server.

        Args:
            server_name: Name of the server to connect to.
        """
        logger.info(f"Connecting to server: {server_name}")

        # Get server configuration
        server_config = self.config.get_server_config(server_name)
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env")

        if not command:
            raise ValueError(f"Invalid server configuration for '{server_name}': missing 'command'")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write_stream = stdio_transport

        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write_stream))
        await session.initialize()

        # Store session
        self.sessions[server_name] = session

        # Get available tools
        response = await session.list_tools()
        self.server_tools[server_name] = response.tools

        logger.info(f"Connected to server '{server_name}' with {len(response.tools)} tools")

    async def connect_to_all_servers(self) -> None:
        """Connect to all configured MCP servers."""
        server_names = self.config.list_servers()
        for server_name in server_names:
            try:
                await self.connect_to_server(server_name)
            except Exception as e:
                logger.error(f"Failed to connect to server '{server_name}': {str(e)}")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get a list of all available tools from connected servers."""
        tools = []
        self.tool_map = {}  # Reset tool map

        for server_name, server_tools in self.server_tools.items():
            # Normalize server name for tool naming (replace hyphens with underscores)
            normalized_server = server_name.replace('-', '_')
            
            for tool in server_tools:
                # Create a consistent tool name with clear separator
                fq_tool_name = f"{normalized_server}{self.TOOL_NAME_SEPARATOR}{tool.name}"
                
                # Store the mapping from full qualified name to (server_name, tool_name)
                self.tool_map[fq_tool_name] = (server_name, tool.name)
                
                tools.append({
                    "name": fq_tool_name,
                    "description": tool.description or f"Tool from {server_name}",
                    "input_schema": tool.inputSchema
                })

        logger.debug(f"Registered tools: {self.tool_map}")
        return tools

    async def call_tool(self, full_tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool on an MCP server.

        Args:
            full_tool_name: Tool name in format "server_name__tool_name"
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        logger.debug(f"Calling tool: {full_tool_name} with args: {arguments}")
        logger.debug(f"Available tool map: {self.tool_map}")
        
        if full_tool_name in self.tool_map:
            # Use the mapping we created in get_available_tools
            server_name, tool_name = self.tool_map[full_tool_name]
        else:
            # Fallback to parsing the name if not in our map (should be avoided)
            try:
                parts = full_tool_name.split(self.TOOL_NAME_SEPARATOR, 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid tool name format: {full_tool_name}")
                
                normalized_server, tool_name = parts
                # Convert back normalized server name to actual server name
                server_name = normalized_server.replace('_', '-')
                
                logger.warning(f"Tool {full_tool_name} not in tool map, parsed as server:{server_name}, tool:{tool_name}")
            except ValueError:
                raise ValueError(f"Invalid tool name format: {full_tool_name}. Expected 'server_name{self.TOOL_NAME_SEPARATOR}tool_name'")

        if server_name not in self.sessions:
            raise ValueError(f"Server '{server_name}' not connected. Available servers: {list(self.sessions.keys())}")

        session = self.sessions[server_name]
        logger.debug(f"Executing {tool_name} on server {server_name}")
        result = await session.call_tool(tool_name, arguments)

        # Convert MCP result to a dictionary for easier processing
        output = {}

        if hasattr(result, "content") and result.content:
            # Handle text content
            texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
            if texts:
                output["text"] = "\n".join(texts)

            # TODO: Handle other types of content if needed, eg images, files, etc.
            # ...

        if hasattr(result, "isError") and result.isError:
            output["error"] = True

        return output

    async def process_query(
        self,
        query: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Process a query using Claude and available tools.

        Args:
            query: User query to process
            system_prompt: Optional system prompt
            temperature: Temperature for model generation
            stream: Whether to stream the response

        Yields:
            Generated text chunks as they become available
        """
        # Ensure we have tools available
        all_tools = self.get_available_tools()

        messages: List[MessageParam] = [
            {
                "role": "user",
                "content": query
            }
        ]
        return self._process_query(messages, system_prompt, temperature, stream, all_tools)

    async def _process_query(self, messages: list, system_prompt: str, temperature: float = 0.7, stream: bool = True, all_tools: list = []) -> AsyncGenerator[str, None]:
        class EventType(str, Enum):
            MESSAGE_START = 'message_start'
            CONTENT_BLOCK_START = 'content_block_start'
            CONTENT_BLOCK_DELTA = 'content_block_delta'
            CONTENT_BLOCK_STOP = 'content_block_stop'
            PING = 'ping'
            ERROR = 'error'
            TOOL_USE = 'tool_use'
            MESSAGE_DELTA = 'message_delta'
            MESSAGE_STOP = 'message_stop'


        class MessageType(str, Enum):
            MESSAGE = 'message'
            TOOL_USE = 'tool_use'

        message_type: MessageType = None
        message_text = ""
        tool_args = ""
        tool_name = None
        tool_id = None
        message_usage = None
        # Start streaming response
        response_generator = await self.async_anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            tools=all_tools,
            stream=stream
        )

        async for event in response_generator:
            if event.type == 'input_json':
                logger.debug(f"delta: {repr(event.partial_json)}")
                logger.debug(f"snapshot: {event.snapshot}")
            elif event.type=='message_start':
                evt: RawMessageStartEvent = event
                if evt.message.type == MessageType.MESSAGE:
                    message_text = ""
                elif evt.message.type == MessageType.TOOL_USE:
                    tool_args = ""
                else:
                    logger.warning(f"unhandled message_start: {evt}")
                message_usage = evt.message.usage

            elif event.type=='content_block_start':
                evt: RawContentBlockStartEvent = event
                if evt.content_block.type == MessageType.MESSAGE:
                    message_text = ""
                elif evt.content_block.type == MessageType.TOOL_USE:
                    tool_name = evt.content_block.name
                    tool_id = evt.content_block.id

            elif event.type=='content_block_stop':
                evt: RawContentBlockStopEvent = event

            elif event.type=='content_block_delta':
                evt: RawContentBlockDeltaEvent = event
                if evt.delta.type == 'text_delta':
                    message_text += evt.delta.text
                    yield evt.delta.text
                elif evt.delta.type == 'input_json_delta':
                    tool_args += evt.delta.partial_json
                else:
                    logger.warning(f"unhandled content_block_delta: {evt}")

            elif event.type =='tool_use':
                logger.debug(f"tool_use: {event.tool_use}")
            elif event.type=='message':
                logger.debug(f"message: {event.message}")
            elif event.type=='message_stop':
                logger.debug(f"message_stop: {event.type}")
                continue
            elif event.type == 'message_delta':
                logger.debug(f"{event.delta.stop_reason} / {event.usage}")
            else:
                logger.warning(f"event not handled: {event}")

        if tool_args:
            import json
            try:
                tool_args_dict = json.loads(tool_args)
            except json.JSONDecodeError:
                logger.error(f"could not decode: {tool_args}")
                tool_args_dict = {}

        if tool_name is not None:
            try:
                logger.debug(f"will call {tool_name} with {tool_args_dict}")
                tool_result = await self.call_tool(tool_name, tool_args_dict)

                # Add tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_args_dict}
                    ]
                })

                result_text = tool_result.get("text", "Tool executed successfully")
                if tool_result.get("error"):
                    result_text = f"Error: {result_text}"

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tool_id, "content": result_text}
                    ]
                })

                yield f"\n[Tool result: {result_text}]\n"
                # Continue the conversation with the tool result
                async for chunk in self._process_query(messages, system_prompt, temperature, stream, all_tools):
                    yield chunk
            except Exception as ex:
                logger.exception(f"Error calling tool: {ex}")
                yield f"\n[Error executing tool {tool_name}: {str(ex)}]\n"


    async def cleanup(self):
        """Clean up resources and connections."""
        await self.exit_stack.aclose()
        logger.debug("Cleaned up MCP client resources")
