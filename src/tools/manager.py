"""
Comprehensive tool manager for the autonomous agent.

This module provides unified management of built-in tools, MCP servers,
and custom tools with easy configuration and dynamic loading.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .interfaces import Tool
from .builtin import get_all_builtin_tools
from .mcp import MCPServerManager, MCPToolWrapper, MCPServerConfig, create_default_mcp_config
from .tool_layer import ToolLayer
from ..config import ToolConfig
from ..ethos.interfaces import EthosFramework

logger = logging.getLogger(__name__)


@dataclass
class ToolManagerConfig:
    """Configuration for the tool manager"""
    enable_builtin_tools: bool = True
    enable_mcp_servers: bool = True
    mcp_config_path: Optional[str] = None
    tool_categories_enabled: Optional[List[str]] = None  # None means all enabled
    safety_level: str = "standard"  # strict, standard, permissive
    max_tools_per_category: int = 50


class ToolManager:
    """Comprehensive tool manager for the autonomous agent"""
    
    def __init__(self, config: ToolManagerConfig, ethos_framework: Optional[EthosFramework] = None):
        self.config = config
        self.ethos_framework = ethos_framework
        
        # Tool storage
        self.builtin_tools: List[Tool] = []
        self.mcp_server_manager: Optional[MCPServerManager] = None
        self.mcp_tools: List[MCPToolWrapper] = []
        
        # Tool organization
        self.tools_by_category: Dict[str, List[Tool]] = {}
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def initialize(self) -> bool:
        """Initialize the tool manager with all configured tools"""
        try:
            self.logger.info("Initializing comprehensive tool manager...")
            
            # Load built-in tools
            if self.config.enable_builtin_tools:
                self._load_builtin_tools()
            
            # Setup MCP servers
            if self.config.enable_mcp_servers:
                self._setup_mcp_integration()
            
            # Organize tools by category
            self._organize_tools_by_category()
            
            # Apply safety filters
            self._apply_safety_filters()
            
            total_tools = len(self.builtin_tools) + len(self.mcp_tools)
            self.logger.info(f"Tool manager initialized with {total_tools} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tool manager: {e}")
            return False
    
    def _load_builtin_tools(self) -> None:
        """Load built-in tools"""
        try:
            all_builtin = get_all_builtin_tools()
            
            for tool in all_builtin:
                # Apply category filter if specified
                if (self.config.tool_categories_enabled and 
                    tool.category not in self.config.tool_categories_enabled):
                    continue
                
                # Apply safety level filter
                if self._check_tool_safety(tool):
                    self.builtin_tools.append(tool)
            
            self.logger.info(f"Loaded {len(self.builtin_tools)} built-in tools")
            
        except Exception as e:
            self.logger.error(f"Failed to load built-in tools: {e}")
    
    def _setup_mcp_integration(self) -> None:
        """Setup MCP server integration"""
        try:
            self.mcp_server_manager = MCPServerManager()
            
            # Load MCP server configurations
            if self.config.mcp_config_path:
                from .mcp.config import load_mcp_config
                servers = load_mcp_config(self.config.mcp_config_path)
            else:
                servers = create_default_mcp_config()
            
            # Register and start servers
            for server_config in servers:
                self.mcp_server_manager.register_server(server_config)
                
                if not server_config.disabled:
                    success = self.mcp_server_manager.start_server(server_config.name)
                    if success:
                        self.logger.info(f"Started MCP server: {server_config.name}")
                    else:
                        self.logger.warning(f"Failed to start MCP server: {server_config.name}")
            
            # Create MCP tool wrappers
            self._create_mcp_tool_wrappers()
            
        except Exception as e:
            self.logger.error(f"Failed to setup MCP integration: {e}")
    
    def _create_mcp_tool_wrappers(self) -> None:
        """Create tool wrappers for all available MCP tools"""
        if not self.mcp_server_manager:
            return
        
        try:
            available_tools = self.mcp_server_manager.get_available_tools()
            
            for tool_schema in available_tools:
                # Apply category filter if specified
                if (self.config.tool_categories_enabled and 
                    "mcp_tools" not in self.config.tool_categories_enabled):
                    continue
                
                mcp_tool = MCPToolWrapper(tool_schema, self.mcp_server_manager)
                self.mcp_tools.append(mcp_tool)
            
            self.logger.info(f"Created {len(self.mcp_tools)} MCP tool wrappers")
            
        except Exception as e:
            self.logger.error(f"Failed to create MCP tool wrappers: {e}")
    
    def _organize_tools_by_category(self) -> None:
        """Organize all tools by category"""
        self.tools_by_category = {}
        
        # Process built-in tools
        for tool in self.builtin_tools:
            category = tool.category
            if category not in self.tools_by_category:
                self.tools_by_category[category] = []
            self.tools_by_category[category].append(tool)
        
        # Process MCP tools
        for tool in self.mcp_tools:
            category = tool.category
            if category not in self.tools_by_category:
                self.tools_by_category[category] = []
            self.tools_by_category[category].append(tool)
        
        # Apply category limits
        for category, tools in self.tools_by_category.items():
            if len(tools) > self.config.max_tools_per_category:
                self.tools_by_category[category] = tools[:self.config.max_tools_per_category]
                self.logger.warning(f"Limited {category} tools to {self.config.max_tools_per_category}")
    
    def _apply_safety_filters(self) -> None:
        """Apply safety filters based on configuration"""
        if self.config.safety_level == "strict":
            # Remove tools with potential safety concerns
            safe_builtin_tools = []
            for tool in self.builtin_tools:
                if tool.category not in ["system_operations"]:  # Example restriction
                    safe_builtin_tools.append(tool)
            self.builtin_tools = safe_builtin_tools
            
            # In strict mode, disable MCP servers entirely
            if self.mcp_server_manager:
                self.mcp_server_manager.shutdown_all()
                self.mcp_tools = []
        
        elif self.config.safety_level == "permissive":
            # Allow all tools
            pass
        
        # Standard safety level allows most tools with warnings
    
    def _check_tool_safety(self, tool: Tool) -> bool:
        """Check if a tool meets safety requirements"""
        if self.ethos_framework:
            # Use ethos framework for validation if available
            try:
                from ..models import ToolCall
                dummy_call = ToolCall(
                    tool_name=tool.name,
                    arguments={},
                    timestamp=0
                )
                is_valid, _ = self.ethos_framework.validate_tool_call(dummy_call)
                return is_valid
            except:
                pass
        
        # Default safety check based on category
        if self.config.safety_level == "strict":
            restricted_categories = ["system_operations"]
            return tool.category not in restricted_categories
        
        return True
    
    def register_with_tool_layer(self, tool_layer: ToolLayer) -> int:
        """Register all managed tools with a ToolLayer instance"""
        registered_count = 0
        
        try:
            # Register built-in tools
            for tool in self.builtin_tools:
                try:
                    tool_layer.register_tool(tool)
                    registered_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to register built-in tool {tool.name}: {e}")
            
            # Register MCP tools
            for tool in self.mcp_tools:
                try:
                    tool_layer.register_tool(tool)
                    registered_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to register MCP tool {tool.name}: {e}")
            
            self.logger.info(f"Registered {registered_count} tools with ToolLayer")
            return registered_count
            
        except Exception as e:
            self.logger.error(f"Failed to register tools with ToolLayer: {e}")
            return registered_count
    
    def get_tool_inventory(self) -> Dict[str, Any]:
        """Get comprehensive inventory of all managed tools"""
        inventory = {
            "builtin_tools": {
                "count": len(self.builtin_tools),
                "tools": {tool.name: {
                    "description": tool.description,
                    "category": tool.category
                } for tool in self.builtin_tools}
            },
            "mcp_tools": {
                "count": len(self.mcp_tools),
                "tools": {tool.name: {
                    "description": tool.description,
                    "category": tool.category,
                    "server": tool.schema.server_name
                } for tool in self.mcp_tools},
                "servers": {}
            },
            "categories": {category: [tool.name for tool in tools] 
                         for category, tools in self.tools_by_category.items()},
            "total_tools": len(self.builtin_tools) + len(self.mcp_tools)
        }
        
        # Add MCP server status
        if self.mcp_server_manager:
            inventory["mcp_tools"]["servers"] = {
                "server_status": self.mcp_server_manager.get_server_status()
            }
        
        return inventory
    
    def get_total_tool_count(self) -> int:
        """Get total count of all managed tools"""
        return len(self.builtin_tools) + len(self.mcp_tools)
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a specific category"""
        return self.tools_by_category.get(category, [])
    
    def get_available_categories(self) -> List[str]:
        """Get list of all available tool categories"""
        return list(self.tools_by_category.keys())
    
    def search_tools(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for tools by name or description"""
        results = []
        query_lower = query.lower()
        
        # Search built-in tools
        for tool in self.builtin_tools:
            if category and tool.category != category:
                continue
            
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append({
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "type": "builtin"
                })
        
        # Search MCP tools
        for tool in self.mcp_tools:
            if category and tool.category != category:
                continue
            
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append({
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "type": "mcp",
                    "server": tool.schema.server_name
                })
        
        return results
    
    def shutdown(self) -> None:
        """Shutdown the tool manager and cleanup resources"""
        try:
            if self.mcp_server_manager:
                self.mcp_server_manager.shutdown_all()
            
            self.logger.info("Tool manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during tool manager shutdown: {e}")


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def create_default_config() -> ToolManagerConfig:
    """Create a default tool manager configuration"""
    return ToolManagerConfig(
        enable_builtin_tools=True,
        enable_mcp_servers=True,
        mcp_config_path=".kiro/settings/mcp.json",
        tool_categories_enabled=None,  # All categories
        safety_level="standard",
        max_tools_per_category=50
    )


def create_research_config() -> ToolManagerConfig:
    """Create configuration focused on research tools"""
    return ToolManagerConfig(
        enable_builtin_tools=True,
        enable_mcp_servers=True,
        tool_categories_enabled=[
            "data_processing", "web_operations", "file_operations", 
            "mathematics", "mcp_tools"
        ],
        safety_level="standard",
        max_tools_per_category=30
    )


def create_creative_config() -> ToolManagerConfig:
    """Create configuration focused on creative tools"""
    return ToolManagerConfig(
        enable_builtin_tools=True,
        enable_mcp_servers=True,
        tool_categories_enabled=[
            "file_operations", "data_processing", "web_operations", "mcp_tools"
        ],
        safety_level="permissive",
        max_tools_per_category=40
    )


def create_secure_config() -> ToolManagerConfig:
    """Create a security-focused configuration"""
    return ToolManagerConfig(
        enable_builtin_tools=True,
        enable_mcp_servers=False,  # Disable MCP for security
        tool_categories_enabled=[
            "data_processing", "security", "mathematics", "file_operations"
        ],
        safety_level="strict",
        max_tools_per_category=20
    )