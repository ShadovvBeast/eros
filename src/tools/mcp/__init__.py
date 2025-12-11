"""
MCP (Model Context Protocol) integration for the Autonomous Agent.

This package provides MCP server management and tool integration capabilities.
"""

from .server_manager import MCPServerManager
from .tool_wrapper import MCPToolWrapper
from .config import MCPServerConfig, create_default_mcp_config

__all__ = [
    'MCPServerManager',
    'MCPToolWrapper', 
    'MCPServerConfig',
    'create_default_mcp_config'
]