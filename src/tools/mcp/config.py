"""
MCP server configuration management.
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    working_directory: Optional[str] = None
    timeout: int = 30
    auto_restart: bool = True
    disabled: bool = False
    description: str = ""
    category: str = "mcp_tools"


def create_default_mcp_config() -> List[MCPServerConfig]:
    """Create default MCP server configurations"""
    servers = []
    
    # File system MCP server
    servers.append(MCPServerConfig(
        name="filesystem",
        command="uvx",
        args=["mcp-server-filesystem", "--base-dir", "."],
        env={"FASTMCP_LOG_LEVEL": "ERROR"},
        description="File system operations server",
        category="file_operations"
    ))
    
    # Git MCP server
    servers.append(MCPServerConfig(
        name="git",
        command="uvx",
        args=["mcp-server-git"],
        env={"FASTMCP_LOG_LEVEL": "ERROR"},
        description="Git version control operations",
        category="development"
    ))
    
    # SQLite MCP server
    servers.append(MCPServerConfig(
        name="sqlite",
        command="uvx",
        args=["mcp-server-sqlite"],
        env={"FASTMCP_LOG_LEVEL": "ERROR"},
        description="SQLite database operations",
        category="data_processing"
    ))
    
    # Time and calendar MCP server
    servers.append(MCPServerConfig(
        name="time",
        command="uvx",
        args=["mcp-server-time"],
        env={"FASTMCP_LOG_LEVEL": "ERROR"},
        description="Time and calendar operations",
        category="utilities"
    ))
    
    return servers


def save_mcp_config(servers: List[MCPServerConfig], config_path: str) -> bool:
    """Save MCP server configurations to file"""
    try:
        config = {
            "mcpServers": {}
        }
        
        for server in servers:
            config["mcpServers"][server.name] = {
                "command": server.command,
                "args": server.args,
                "env": server.env,
                "disabled": server.disabled
            }
            
            if server.working_directory:
                config["mcpServers"][server.name]["cwd"] = server.working_directory
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception:
        return False


def load_mcp_config(config_path: str) -> List[MCPServerConfig]:
    """Load MCP server configurations from file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        servers = []
        mcp_servers = config.get("mcpServers", {})
        
        for name, server_config in mcp_servers.items():
            server = MCPServerConfig(
                name=name,
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env", {}),
                working_directory=server_config.get("cwd"),
                disabled=server_config.get("disabled", False),
                description=f"MCP server: {name}"
            )
            servers.append(server)
        
        return servers
        
    except Exception:
        return []