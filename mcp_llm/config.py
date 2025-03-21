import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

class MCPConfig:
    """Configuration manager for MCP servers."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, will look for config in default locations.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.loaded = False
    
    def load_config(self) -> Dict[str, Any]:
        """Load the MCP server configuration from the specified file or default locations."""
        if self.loaded:
            return self.config
            
        # If config_path is provided, try to load from there
        if self.config_path and os.path.isfile(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.loaded = True
                return self.config
            except Exception as e:
                raise ValueError(f"Error loading config from {self.config_path}: {str(e)}")
        
        # Default locations to check
        default_locations = [
            Path.home() / ".config/mcp-client/config.json",
            Path.home() / ".mcp-client.json",
            Path.cwd() / "mcp-client.json",
            Path.cwd() / "config.json",
        ]
        
        # Try each location
        for location in default_locations:
            if location.is_file():
                try:
                    with open(location, 'r') as f:
                        self.config = json.load(f)
                    self.loaded = True
                    self.config_path = str(location)
                    return self.config
                except Exception:
                    continue
        
        # If we got here, no valid config was found
        if not self.loaded:
            config_files = '\n'.join(str(loc) for loc in default_locations)
            raise FileNotFoundError(
                "MCP configuration not found. Please create a config file at one of the following locations:\n"
                f"{config_files}\n"
                "Or specify a config file path with --config"
            )
        
        return self.config
    
    def get_server_config(self, server_name: str) -> Dict[str, Any]:
        """Get configuration for a specific server."""
        if not self.loaded:
            self.load_config()
        
        if "mcpServers" not in self.config:
            raise ValueError("Invalid config: 'mcpServers' key not found")
        
        servers = self.config.get("mcpServers", {})
        if server_name not in servers:
            raise ValueError(f"Server '{server_name}' not found in configuration")
        
        return servers[server_name]
    
    def list_servers(self) -> list[str]:
        """Get a list of all configured server names."""
        if not self.loaded:
            self.load_config()
        
        return list(self.config.get("mcpServers", {}).keys())
