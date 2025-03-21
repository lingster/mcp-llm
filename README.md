# MCP Client

A CLI tool for interacting with Anthropic's Claude model using MCP (Model Context Protocol) servers.

## Features

- Connect to multiple MCP servers defined in a configuration file
- Interactive chat with Claude using available MCP tools
- Stream Claude's responses in real-time
- Support for both interactive and single-query modes
- Rich terminal output with clear formatting

## Installation

```bash
# Clone the repository
git clone https://github.com/lingster/mcp-llm.git
cd mcp-llm

# Install with uv
uv sync 

```

## Configuration

Create a JSON configuration file that defines your MCP servers. For example:

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Save this file to one of these locations:
- `~/.config/mcp-client/config.json`
- `~/.mcp-client.json`
- `./mcp-client.json`
- `./config.json`

Alternatively, you can specify a custom config file with the `--config` option.

## Usage

### Interactive Chat

```bash
# Start interactive chat with all configured servers
uv run mcpllm chat

# Connect to specific servers only
uv run mcpllm chat --server brave-search 

# Use a custom configuration file
uv run mcpllm chat --config /path/to/config.json

# Customize Claude model and parameters
uv run mcpllm chat --model claude-3-5-sonnet-20241022 --temperature 0.8 --max-tokens 8192
```

### Single-Query Mode

```bash
# Process a single query and exit
uv run mcpllm chat --query "Is it raining in London?"
```

```bash
# tool use
uv run mcpllm chat --server brave-search --query "Where's wally?"
```

### List Configured Servers

```bash
# Show all configured servers
mcpllm servers
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)

You can also create a `.env` file in your current directory:

```
ANTHROPIC_API_KEY=your-api-key-here
```

## License

MIT
