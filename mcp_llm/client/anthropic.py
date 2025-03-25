from enum import Enum
import os
import sys
import json
from loguru import logger
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple, Union, cast, Iterable, TypeVar, Protocol, runtime_checkable

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (
    Message, MessageParam,
    RawMessageStartEvent, RawContentBlockStartEvent, RawMessageStopEvent, 
    RawMessageDeltaEvent, RawContentBlockStopEvent, RawContentBlockDeltaEvent,
    MessageCreateParams
)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Resource, Tool

from ..config import MCPConfig

logger.remove()
logger.add(sys.stderr, level="INFO")


# Define ToolParam since it's missing from anthropic.types
class ToolParam(Dict[str, Any]):
    """Tool parameter structure for Anthropic API."""
    pass


class EventType(str, Enum):
    """Event types from Anthropic streaming API."""
    MESSAGE_START = 'message_start'
    CONTENT_BLOCK_START = 'content_block_start'
    CONTENT_BLOCK_DELTA = 'content_block_delta'
    CONTENT_BLOCK_STOP = 'content_block_stop'
    PING = 'ping'
    ERROR = 'error'
    TOOL_USE = 'tool_use'
    MESSAGE_DELTA = 'message_delta'
    MESSAGE_STOP = 'message_stop'


class ContentType(str, Enum):
    """Content types from Anthropic API."""
    TEXT = 'text'
    TOOL_USE = 'tool_use'


@runtime_checkable
class ContentBlock(Protocol):
    """Protocol for content blocks."""
    type: str


@runtime_checkable
class TextContentBlock(ContentBlock, Protocol):
    """Protocol for text content blocks."""
    type: str
    text: str


@runtime_checkable
class ToolUseContentBlock(ContentBlock, Protocol):
    """Protocol for tool use content blocks."""
    type: str
    name: str
    id: str
    input: Dict[str, Any]


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
        output: Dict[str, Any] = {}

        if hasattr(result, "content") and result.content:
            # Handle text content by checking each content item
            text_chunks = []
            for content in result.content:
                # Safe check for text attribute
                if hasattr(content, "text") and content.text:
                    text_chunks.append(content.text)
            
            if text_chunks:
                output["text"] = "\n".join(text_chunks)

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
        async for chunk in self._process_query(messages, system_prompt, temperature, stream, all_tools):
            yield chunk

    async def _process_query(
        self, 
        messages: List[MessageParam], 
        system_prompt: str, 
        temperature: float = 0.7, 
        stream: bool = True, 
        all_tools: List[Dict[str, Any]] = []
    ) -> AsyncGenerator[str, None]:
        """Internal method to process queries with streaming support.
        
        Args:
            messages: List of messages to send to Claude
            system_prompt: System prompt for Claude
            temperature: Temperature for model generation
            stream: Whether to stream the response
            all_tools: List of available tools
            
        Yields:
            Generated text chunks as they become available
        """
        message_text = ""
        tool_args = ""
        tool_name = None
        tool_id = None
        content_type = None

        # Create kwargs to handle the API call safely for type checking
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages,
            "stream": stream
        }
        
        # Use this pattern to bypass type checking issues
        # Only add tools if we have them
        if all_tools:
            kwargs["tools"] = [{
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            } for tool in all_tools]
        
        # Start streaming response
        response = await self.async_anthropic.messages.create(**kwargs)

        if not stream:
            # For non-streaming response, directly process the message content
            if hasattr(response, "content"):
                has_tool_call = False
                
                for content_block in response.content:
                    if isinstance(content_block, dict) and "type" in content_block:
                        # Handle different types of content blocks
                        if content_block["type"] == "text":
                            if "text" in content_block:
                                yield content_block["text"]
                        elif content_block["type"] == "tool_use":
                            has_tool_call = True
                            tool_name = content_block.get("name")
                            tool_id = content_block.get("id")
                            tool_args = json.dumps(content_block.get("input", {}))
                    else:
                        # Try direct attribute access if not a dict
                        if hasattr(content_block, "type"):
                            if content_block.type == "text" and hasattr(content_block, "text"):
                                yield content_block.text
                            elif content_block.type == "tool_use":
                                has_tool_call = True
                                if hasattr(content_block, "name"):
                                    tool_name = content_block.name
                                if hasattr(content_block, "id"):
                                    tool_id = content_block.id
                                if hasattr(content_block, "input"):
                                    tool_args = json.dumps(content_block.input)
                
                if has_tool_call and tool_name and tool_id and tool_args:
                    try:
                        tool_args_dict = json.loads(tool_args)
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
            
            return
            
        # Streaming response handling
        try:
            # Safe streaming using iterator
            if hasattr(response, "__aiter__"):
                async for event in response:
                    if hasattr(event, "type"):
                        event_type = event.type
                        
                        if event_type == 'input_json':
                            # Safely handle input_json events
                            if hasattr(event, 'partial_json'):
                                logger.debug(f"delta: {repr(event.partial_json)}")
                            if hasattr(event, 'snapshot'):
                                logger.debug(f"snapshot: {event.snapshot}")
                        
                        elif event_type == 'message_start':
                            # Message start event
                            if hasattr(event, "message") and hasattr(event.message, "type"):
                                if event.message.type == 'message':
                                    message_text = ""
                                    content_type = ContentType.TEXT
                                elif event.message.type == 'tool_use':
                                    tool_args = ""
                                    content_type = ContentType.TOOL_USE
                                else:
                                    logger.warning(f"Unhandled message_start type: {event.message.type}")
                        
                        elif event_type == 'content_block_start':
                            # Content block start event
                            if hasattr(event, "content_block") and hasattr(event.content_block, "type"):
                                if event.content_block.type == 'text':
                                    message_text = ""
                                elif event.content_block.type == 'tool_use':
                                    if hasattr(event.content_block, "name"):
                                        tool_name = event.content_block.name
                                    if hasattr(event.content_block, "id"):
                                        tool_id = event.content_block.id
                        
                        elif event_type == 'content_block_delta':
                            # Content delta event
                            if hasattr(event, "delta") and hasattr(event.delta, "type"):
                                if event.delta.type == 'text_delta':
                                    if hasattr(event.delta, "text"):
                                        message_text += event.delta.text
                                        yield event.delta.text
                                elif event.delta.type == 'input_json_delta':
                                    if hasattr(event.delta, "partial_json"):
                                        tool_args += event.delta.partial_json
                                else:
                                    logger.warning(f"Unhandled content_block_delta: {event.delta.type}")
                        
                        elif event_type == 'tool_use':
                            # Tool use event - just log it
                            logger.debug(f"tool_use event received")
                        
                        elif event_type == 'message_delta':
                            # Message delta event - log details
                            logger.debug(f"Message delta received")
                        
                        elif event_type == 'message_stop':
                            # Message stop event
                            logger.debug("Received message_stop event")
                        
                        elif event_type == 'content_block_stop':
                            # Content block stop event
                            logger.debug("Received content_block_stop event")
                        
                        else:
                            logger.warning(f"Event not handled: {event_type}")
                    else:
                        logger.warning(f"Event has no type attribute: {event}")
            else:
                # Handle non-async-iterable response
                yield "Error: Streaming response is not in expected format"
                return
        
        except Exception as e:
            logger.exception(f"Error in stream processing: {e}")
            yield f"\n[Error: {str(e)}]\n"
            
        # Process tool if we received one
        if tool_args and tool_name is not None and tool_id is not None:
            try:
                # Parse tool arguments
                try:
                    tool_args_dict = json.loads(tool_args)
                except json.JSONDecodeError:
                    logger.error(f"Could not decode tool arguments: {tool_args}")
                    tool_args_dict = {}
                
                logger.debug(f"Will call {tool_name} with {tool_args_dict}")
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
        
    async def close(self):
        """Alias for cleanup() for compatibility."""
        await self.cleanup()
        
    async def aclose(self):
        """Alias for cleanup() for compatibility."""
        await self.cleanup()
