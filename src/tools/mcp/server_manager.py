"""
MCP server management and communication.
"""

import json
import subprocess
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPToolSchema:
    """Schema definition for an MCP tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


class MCPServerManager:
    """Manages MCP server instances and communication"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.server_tools: Dict[str, List[MCPToolSchema]] = {}
        self.connection_status: Dict[str, bool] = {}
        
    def register_server(self, config: MCPServerConfig) -> bool:
        """Register a new MCP server configuration"""
        try:
            self.servers[config.name] = config
            self.connection_status[config.name] = False
            logger.info(f"Registered MCP server: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register MCP server {config.name}: {e}")
            return False
    
    def start_server(self, server_name: str) -> bool:
        """Start an MCP server process"""
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not registered")
            return False
        
        config = self.servers[server_name]
        
        if config.disabled:
            logger.info(f"Server {server_name} is disabled")
            return False
        
        try:
            # Prepare environment
            import os
            env = os.environ.copy()
            env.update(config.env)
            
            # Start the server process
            process = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=config.working_directory,
                text=True,
                bufsize=0
            )
            
            self.server_processes[server_name] = process
            self.connection_status[server_name] = True
            
            # Initialize server communication
            self._initialize_server_communication(server_name)
            
            logger.info(f"Started MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server {server_name}: {e}")
            self.connection_status[server_name] = False
            return False
    
    def stop_server(self, server_name: str) -> bool:
        """Stop an MCP server process"""
        if server_name not in self.server_processes:
            return True
        
        try:
            process = self.server_processes[server_name]
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            del self.server_processes[server_name]
            self.connection_status[server_name] = False
            
            logger.info(f"Stopped MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCP server {server_name}: {e}")
            return False
    
    def _initialize_server_communication(self, server_name: str) -> None:
        """Initialize communication with an MCP server"""
        try:
            # Send initialization message
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "autonomous-agent",
                        "version": "1.0.0"
                    }
                }
            }
            
            self._send_message(server_name, init_message)
            
            # Get available tools
            self._discover_server_tools(server_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize communication with {server_name}: {e}")
    
    def _discover_server_tools(self, server_name: str) -> None:
        """Discover available tools from an MCP server"""
        try:
            # Request tools list
            tools_message = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            response = self._send_message(server_name, tools_message)
            
            if response and "result" in response:
                tools_data = response["result"].get("tools", [])
                server_tools = []
                
                for tool_data in tools_data:
                    tool_schema = MCPToolSchema(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        server_name=server_name
                    )
                    server_tools.append(tool_schema)
                
                self.server_tools[server_name] = server_tools
                logger.info(f"Discovered {len(server_tools)} tools from {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to discover tools from {server_name}: {e}")
    
    def _send_message(self, server_name: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a message to an MCP server and get response"""
        if server_name not in self.server_processes:
            return None
        
        try:
            process = self.server_processes[server_name]
            
            # Send message
            message_str = json.dumps(message) + "\n"
            process.stdin.write(message_str)
            process.stdin.flush()
            
            # Read response
            response_str = process.stdout.readline()
            if response_str:
                return json.loads(response_str.strip())
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to send message to {server_name}: {e}")
            return None
    
    def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on an MCP server"""
        if server_name not in self.server_processes:
            return {"success": False, "error": f"Server {server_name} not running"}
        
        try:
            # Prepare tool call message
            tool_message = {
                "jsonrpc": "2.0",
                "id": int(time.time()),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = self._send_message(server_name, tool_message)
            
            if response:
                if "result" in response:
                    return {
                        "success": True,
                        "result": response["result"],
                        "server": server_name,
                        "tool": tool_name
                    }
                elif "error" in response:
                    return {
                        "success": False,
                        "error": response["error"].get("message", "Unknown error"),
                        "server": server_name,
                        "tool": tool_name
                    }
            
            return {"success": False, "error": "No response from server"}
            
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name} on {server_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered servers"""
        status = {}
        
        for server_name, config in self.servers.items():
            is_running = server_name in self.server_processes
            tool_count = len(self.server_tools.get(server_name, []))
            
            status[server_name] = {
                "running": is_running,
                "connected": self.connection_status.get(server_name, False),
                "disabled": config.disabled,
                "tool_count": tool_count,
                "description": config.description,
                "category": config.category
            }
        
        return status
    
    def get_available_tools(self) -> List[MCPToolSchema]:
        """Get all available tools from all servers"""
        all_tools = []
        
        for server_name, tools in self.server_tools.items():
            if self.connection_status.get(server_name, False):
                all_tools.extend(tools)
        
        return all_tools
    
    def shutdown_all(self) -> None:
        """Shutdown all MCP servers"""
        for server_name in list(self.server_processes.keys()):
            self.stop_server(server_name)