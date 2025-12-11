"""
MCP (Model Context Protocol) Server Integration

This module provides comprehensive MCP server support for the autonomous agent,
allowing easy integration with external MCP servers and tools.
"""

import json
import asyncio
import subprocess
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile
import os
import sys

# Import the tool interfaces
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from src.tools.interfaces import Tool
from src.models import ToolResult

logger = logging.getLogger(__name__)


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
    
    def restart_server(self, server_name: str) -> bool:
        """Restart an MCP server"""
        self.stop_server(server_name)
        time.sleep(1)  # Brief pause
        return self.start_server(server_name)
    
    def shutdown_all(self) -> None:
        """Shutdown all MCP servers"""
        for server_name in list(self.server_processes.keys()):
            self.stop_server(server_name)


class MCPTool(Tool):
    """Wrapper for MCP server tools"""
    
    def __init__(self, schema: MCPToolSchema, server_manager: MCPServerManager):
        self.schema = schema
        self.server_manager = server_manager
    
    @property
    def name(self) -> str:
        return f"{self.schema.server_name}.{self.schema.name}"
    
    @property
    def description(self) -> str:
        return f"[MCP:{self.schema.server_name}] {self.schema.description}"
    
    @property
    def category(self) -> str:
        return "mcp_tools"
    
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the MCP tool"""
        try:
            result = self.server_manager.execute_tool(
                self.schema.server_name,
                self.schema.name,
                arguments
            )
            
            return ToolResult(
                tool_name=self.name,
                result=result,
                success=result.get("success", False),
                error_message=result.get("error") if not result.get("success", False) else None,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                result=None,
                success=False,
                error_message=str(e),
                timestamp=int(time.time())
            )
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate arguments against the tool's input schema"""
        try:
            # Basic validation against schema
            schema = self.schema.input_schema
            
            if "properties" in schema:
                required = schema.get("required", [])
                
                # Check required fields
                for field in required:
                    if field not in arguments:
                        return False, f"Missing required field: {field}"
                
                # Check field types (basic validation)
                for field, value in arguments.items():
                    if field in schema["properties"]:
                        expected_type = schema["properties"][field].get("type")
                        if expected_type:
                            if expected_type == "string" and not isinstance(value, str):
                                return False, f"Field {field} must be a string"
                            elif expected_type == "integer" and not isinstance(value, int):
                                return False, f"Field {field} must be an integer"
                            elif expected_type == "number" and not isinstance(value, (int, float)):
                                return False, f"Field {field} must be a number"
                            elif expected_type == "boolean" and not isinstance(value, bool):
                                return False, f"Field {field} must be a boolean"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"


# ============================================================================
# PREDEFINED MCP SERVER CONFIGURATIONS
# ============================================================================

def get_default_mcp_servers() -> List[MCPServerConfig]:
    """Get default MCP server configurations"""
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
    
    # Web search MCP server (if available)
    servers.append(MCPServerConfig(
        name="web_search",
        command="uvx",
        args=["mcp-server-brave-search"],
        env={
            "FASTMCP_LOG_LEVEL": "ERROR",
            "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")
        },
        description="Web search capabilities",
        category="web_operations",
        disabled=not os.getenv("BRAVE_API_KEY")  # Disable if no API key
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


def create_mcp_config_file(servers: List[MCPServerConfig], config_path: str) -> bool:
    """Create an MCP configuration file"""
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
        
        logger.info(f"Created MCP configuration file: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create MCP config file: {e}")
        return False


def load_mcp_config_file(config_path: str) -> List[MCPServerConfig]:
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
        
        logger.info(f"Loaded {len(servers)} MCP server configurations")
        return servers
        
    except Exception as e:
        logger.error(f"Failed to load MCP config file: {e}")
        return []


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

class MCPIntegration:
    """Main integration class for MCP servers with the agent"""
    
    def __init__(self):
        self.server_manager = MCPServerManager()
        self.mcp_tools: List[MCPTool] = []
    
    def setup_default_servers(self) -> bool:
        """Set up default MCP servers"""
        try:
            default_servers = get_default_mcp_servers()
            
            for server_config in default_servers:
                self.server_manager.register_server(server_config)
                
                if not server_config.disabled:
                    success = self.server_manager.start_server(server_config.name)
                    if success:
                        logger.info(f"Started MCP server: {server_config.name}")
                    else:
                        logger.warning(f"Failed to start MCP server: {server_config.name}")
            
            # Create MCP tools from available servers
            self._create_mcp_tools()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup default MCP servers: {e}")
            return False
    
    def setup_from_config(self, config_path: str) -> bool:
        """Set up MCP servers from configuration file"""
        try:
            servers = load_mcp_config_file(config_path)
            
            for server_config in servers:
                self.server_manager.register_server(server_config)
                
                if not server_config.disabled:
                    self.server_manager.start_server(server_config.name)
            
            self._create_mcp_tools()
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup MCP servers from config: {e}")
            return False
    
    def _create_mcp_tools(self) -> None:
        """Create tool wrappers for all available MCP tools"""
        self.mcp_tools = []
        
        available_tools = self.server_manager.get_available_tools()
        
        for tool_schema in available_tools:
            mcp_tool = MCPTool(tool_schema, self.server_manager)
            self.mcp_tools.append(mcp_tool)
        
        logger.info(f"Created {len(self.mcp_tools)} MCP tool wrappers")
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all available MCP tools"""
        return self.mcp_tools
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive status of MCP integration"""
        return {
            "server_status": self.server_manager.get_server_status(),
            "total_tools": len(self.mcp_tools),
            "tools_by_server": {
                server_name: len(tools) 
                for server_name, tools in self.server_manager.server_tools.items()
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown all MCP servers"""
        self.server_manager.shutdown_all()


if __name__ == "__main__":
    # Demo MCP integration
    print("MCP Integration Demo")
    print("=" * 30)
    
    integration = MCPIntegration()
    
    # Setup default servers
    print("Setting up default MCP servers...")
    success = integration.setup_default_servers()
    
    if success:
        print("MCP servers setup completed!")
        
        # Show status
        status = integration.get_server_status()
        print(f"\nServer Status:")
        for server, info in status["server_status"].items():
            print(f"  {server}: {'Running' if info['running'] else 'Stopped'} "
                  f"({info['tool_count']} tools)")
        
        print(f"\nTotal MCP tools available: {status['total_tools']}")
        
        # List some tools
        tools = integration.get_all_tools()
        if tools:
            print("\nSample MCP tools:")
            for tool in tools[:5]:  # Show first 5
                print(f"  - {tool.name}: {tool.description}")
    
    else:
        print("Failed to setup MCP servers")
    
    # Cleanup
    integration.shutdown()