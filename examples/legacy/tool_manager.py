"""
Comprehensive Tool Manager

This module provides a unified interface for managing both classic tools
and MCP servers, with easy configuration and dynamic tool loading.
"""

import json
import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union, Type
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util
import inspect

# Import our tool systems
from comprehensive_toolkit import get_comprehensive_toolkit, ComprehensiveTool
from mcp_integration import MCPIntegration, MCPServerConfig

# Import the base interfaces
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from src.tools.interfaces import Tool
from src.tools.tool_layer import ToolLayer, MCPTool as BaseMCPTool
from src.config import ToolConfig
from src.ethos.interfaces import EthosFramework

logger = logging.getLogger(__name__)


@dataclass
class ToolManagerConfig:
    """Configuration for the tool manager"""
    enable_comprehensive_toolkit: bool = True
    enable_mcp_servers: bool = True
    mcp_config_path: Optional[str] = None
    custom_tool_directories: List[str] = None
    auto_discover_tools: bool = True
    tool_categories_enabled: List[str] = None  # None means all enabled
    safety_level: str = "standard"  # strict, standard, permissive
    max_tools_per_category: int = 50
    enable_tool_caching: bool = True


class ToolManager:
    """Comprehensive tool manager for the autonomous agent"""
    
    def __init__(self, config: ToolManagerConfig, ethos_framework: Optional[EthosFramework] = None):
        self.config = config
        self.ethos_framework = ethos_framework
        
        # Tool storage
        self.classic_tools: Dict[str, Tool] = {}
        self.mcp_integration: Optional[MCPIntegration] = None
        self.custom_tools: Dict[str, Tool] = {}
        
        # Tool metadata and statistics
        self.tool_categories: Dict[str, List[str]] = {}
        self.tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        self.tool_performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def initialize(self) -> bool:
        """Initialize the tool manager with all configured tools"""
        try:
            self.logger.info("Initializing comprehensive tool manager...")
            
            # Load comprehensive toolkit
            if self.config.enable_comprehensive_toolkit:
                self._load_comprehensive_toolkit()
            
            # Setup MCP servers
            if self.config.enable_mcp_servers:
                self._setup_mcp_integration()
            
            # Load custom tools
            if self.config.custom_tool_directories:
                self._load_custom_tools()
            
            # Auto-discover tools if enabled
            if self.config.auto_discover_tools:
                self._auto_discover_tools()
            
            # Organize tools by category
            self._organize_tools_by_category()
            
            # Apply safety filters
            self._apply_safety_filters()
            
            self.logger.info(f"Tool manager initialized with {self.get_total_tool_count()} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tool manager: {e}")
            return False
    
    def _load_comprehensive_toolkit(self) -> None:
        """Load the comprehensive toolkit"""
        try:
            comprehensive_tools = get_comprehensive_toolkit()
            
            for tool in comprehensive_tools:
                # Apply category filter if specified
                if (self.config.tool_categories_enabled and 
                    tool.category not in self.config.tool_categories_enabled):
                    continue
                
                # Apply safety level filter
                if not self._check_tool_safety(tool):
                    continue
                
                self.classic_tools[tool.name] = tool
                self.logger.debug(f"Loaded comprehensive tool: {tool.name}")
            
            self.logger.info(f"Loaded {len(self.classic_tools)} tools from comprehensive toolkit")
            
        except Exception as e:
            self.logger.error(f"Failed to load comprehensive toolkit: {e}")
    
    def _setup_mcp_integration(self) -> None:
        """Setup MCP server integration"""
        try:
            self.mcp_integration = MCPIntegration()
            
            if self.config.mcp_config_path and os.path.exists(self.config.mcp_config_path):
                # Load from config file
                success = self.mcp_integration.setup_from_config(self.config.mcp_config_path)
            else:
                # Use default servers
                success = self.mcp_integration.setup_default_servers()
            
            if success:
                mcp_tools = self.mcp_integration.get_all_tools()
                self.logger.info(f"MCP integration setup with {len(mcp_tools)} tools")
            else:
                self.logger.warning("MCP integration setup failed")
                
        except Exception as e:
            self.logger.error(f"Failed to setup MCP integration: {e}")
    
    def _load_custom_tools(self) -> None:
        """Load custom tools from specified directories"""
        for directory in self.config.custom_tool_directories:
            try:
                self._load_tools_from_directory(directory)
            except Exception as e:
                self.logger.error(f"Failed to load tools from {directory}: {e}")
    
    def _load_tools_from_directory(self, directory: str) -> None:
        """Load tools from a specific directory"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            self.logger.warning(f"Tool directory does not exist: {directory}")
            return
        
        # Look for Python files
        for py_file in directory_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                # Load module dynamically
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for tool classes or functions
                self._extract_tools_from_module(module, py_file.stem)
                
            except Exception as e:
                self.logger.error(f"Failed to load tools from {py_file}: {e}")
    
    def _extract_tools_from_module(self, module, module_name: str) -> None:
        """Extract tools from a loaded module"""
        # Look for Tool classes
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, Tool) and 
                obj != Tool):
                
                try:
                    # Instantiate the tool
                    tool_instance = obj()
                    self.custom_tools[f"{module_name}.{tool_instance.name}"] = tool_instance
                    self.logger.debug(f"Loaded custom tool: {tool_instance.name}")
                except Exception as e:
                    self.logger.error(f"Failed to instantiate tool {name}: {e}")
        
        # Look for tool factory functions
        for name, obj in inspect.getmembers(module):
            if (inspect.isfunction(obj) and 
                name.startswith("create_") and 
                name.endswith("_tools")):
                
                try:
                    tools = obj()
                    if isinstance(tools, list):
                        for tool in tools:
                            if isinstance(tool, Tool):
                                self.custom_tools[f"{module_name}.{tool.name}"] = tool
                                self.logger.debug(f"Loaded custom tool: {tool.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load tools from function {name}: {e}")
    
    def _auto_discover_tools(self) -> None:
        """Auto-discover tools in common locations"""
        discovery_paths = [
            "tools",
            "custom_tools", 
            "agent_tools",
            os.path.expanduser("~/.agent_tools"),
            "/usr/local/share/agent_tools"
        ]
        
        for path in discovery_paths:
            if os.path.exists(path):
                self.logger.info(f"Auto-discovering tools in: {path}")
                self._load_tools_from_directory(path)
    
    def _organize_tools_by_category(self) -> None:
        """Organize all tools by category"""
        self.tool_categories = {}
        
        # Process classic tools
        for tool_name, tool in self.classic_tools.items():
            category = tool.category
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool_name)
        
        # Process MCP tools
        if self.mcp_integration:
            mcp_tools = self.mcp_integration.get_all_tools()
            for tool in mcp_tools:
                category = tool.category
                if category not in self.tool_categories:
                    self.tool_categories[category] = []
                self.tool_categories[category].append(tool.name)
        
        # Process custom tools
        for tool_name, tool in self.custom_tools.items():
            category = tool.category
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool_name)
        
        # Apply category limits
        for category, tools in self.tool_categories.items():
            if len(tools) > self.config.max_tools_per_category:
                self.tool_categories[category] = tools[:self.config.max_tools_per_category]
                self.logger.warning(f"Limited {category} tools to {self.config.max_tools_per_category}")
    
    def _apply_safety_filters(self) -> None:
        """Apply safety filters based on configuration"""
        if self.config.safety_level == "strict":
            # Remove tools with safety concerns
            tools_to_remove = []
            
            for tool_name, tool in self.classic_tools.items():
                if hasattr(tool, 'metadata') and tool.metadata.safety_level == "restricted":
                    tools_to_remove.append(tool_name)
            
            for tool_name in tools_to_remove:
                del self.classic_tools[tool_name]
                self.logger.info(f"Removed restricted tool: {tool_name}")
        
        elif self.config.safety_level == "permissive":
            # Allow all tools
            pass
        
        # Standard safety level allows most tools with warnings
    
    def _check_tool_safety(self, tool: Tool) -> bool:
        """Check if a tool meets safety requirements"""
        if self.ethos_framework:
            # Use ethos framework for validation if available
            try:
                from src.models import ToolCall
                dummy_call = ToolCall(
                    tool_name=tool.name,
                    arguments={},
                    timestamp=0
                )
                is_valid, _ = self.ethos_framework.validate_tool_call(dummy_call)
                return is_valid
            except:
                pass
        
        # Default safety check
        if hasattr(tool, 'metadata'):
            safety_level = getattr(tool.metadata, 'safety_level', 'safe')
            
            if self.config.safety_level == "strict" and safety_level == "restricted":
                return False
            elif self.config.safety_level == "standard" and safety_level == "restricted":
                return False
        
        return True
    
    def register_with_tool_layer(self, tool_layer: ToolLayer) -> int:
        """Register all managed tools with a ToolLayer instance"""
        registered_count = 0
        
        try:
            # Register classic tools
            for tool_name, tool in self.classic_tools.items():
                try:
                    tool_layer.register_tool(tool)
                    registered_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to register classic tool {tool_name}: {e}")
            
            # Register MCP tools
            if self.mcp_integration:
                mcp_tools = self.mcp_integration.get_all_tools()
                for tool in mcp_tools:
                    try:
                        tool_layer.register_tool(tool)
                        registered_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to register MCP tool {tool.name}: {e}")
            
            # Register custom tools
            for tool_name, tool in self.custom_tools.items():
                try:
                    tool_layer.register_tool(tool)
                    registered_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to register custom tool {tool_name}: {e}")
            
            self.logger.info(f"Registered {registered_count} tools with ToolLayer")
            return registered_count
            
        except Exception as e:
            self.logger.error(f"Failed to register tools with ToolLayer: {e}")
            return registered_count
    
    def get_tool_inventory(self) -> Dict[str, Any]:
        """Get comprehensive inventory of all managed tools"""
        inventory = {
            "classic_tools": {
                "count": len(self.classic_tools),
                "tools": {name: {
                    "description": tool.description,
                    "category": tool.category
                } for name, tool in self.classic_tools.items()}
            },
            "mcp_tools": {
                "count": 0,
                "tools": {},
                "servers": {}
            },
            "custom_tools": {
                "count": len(self.custom_tools),
                "tools": {name: {
                    "description": tool.description,
                    "category": tool.category
                } for name, tool in self.custom_tools.items()}
            },
            "categories": self.tool_categories,
            "total_tools": self.get_total_tool_count()
        }
        
        # Add MCP tool information
        if self.mcp_integration:
            mcp_tools = self.mcp_integration.get_all_tools()
            inventory["mcp_tools"]["count"] = len(mcp_tools)
            inventory["mcp_tools"]["tools"] = {
                tool.name: {
                    "description": tool.description,
                    "category": tool.category,
                    "server": tool.schema.server_name
                } for tool in mcp_tools
            }
            inventory["mcp_tools"]["servers"] = self.mcp_integration.get_server_status()
        
        return inventory
    
    def get_total_tool_count(self) -> int:
        """Get total count of all managed tools"""
        count = len(self.classic_tools) + len(self.custom_tools)
        
        if self.mcp_integration:
            count += len(self.mcp_integration.get_all_tools())
        
        return count
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tool names in a specific category"""
        return self.tool_categories.get(category, [])
    
    def get_available_categories(self) -> List[str]:
        """Get list of all available tool categories"""
        return list(self.tool_categories.keys())
    
    def search_tools(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for tools by name or description"""
        results = []
        query_lower = query.lower()
        
        # Search classic tools
        for tool_name, tool in self.classic_tools.items():
            if category and tool.category != category:
                continue
            
            if (query_lower in tool_name.lower() or 
                query_lower in tool.description.lower()):
                results.append({
                    "name": tool_name,
                    "description": tool.description,
                    "category": tool.category,
                    "type": "classic"
                })
        
        # Search MCP tools
        if self.mcp_integration:
            mcp_tools = self.mcp_integration.get_all_tools()
            for tool in mcp_tools:
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
        
        # Search custom tools
        for tool_name, tool in self.custom_tools.items():
            if category and tool.category != category:
                continue
            
            if (query_lower in tool_name.lower() or 
                query_lower in tool.description.lower()):
                results.append({
                    "name": tool_name,
                    "description": tool.description,
                    "category": tool.category,
                    "type": "custom"
                })
        
        return results
    
    def export_tool_configuration(self, output_path: str) -> bool:
        """Export current tool configuration to a file"""
        try:
            config_data = {
                "tool_manager_config": asdict(self.config),
                "tool_inventory": self.get_tool_inventory(),
                "export_timestamp": int(time.time())
            }
            
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Exported tool configuration to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export tool configuration: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the tool manager and cleanup resources"""
        try:
            if self.mcp_integration:
                self.mcp_integration.shutdown()
            
            self.logger.info("Tool manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during tool manager shutdown: {e}")


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def create_default_tool_config() -> ToolManagerConfig:
    """Create a default tool manager configuration"""
    return ToolManagerConfig(
        enable_comprehensive_toolkit=True,
        enable_mcp_servers=True,
        mcp_config_path=".kiro/settings/mcp.json",
        custom_tool_directories=["tools", "custom_tools"],
        auto_discover_tools=True,
        tool_categories_enabled=None,  # All categories
        safety_level="standard",
        max_tools_per_category=50,
        enable_tool_caching=True
    )


def create_research_focused_config() -> ToolManagerConfig:
    """Create configuration focused on research tools"""
    return ToolManagerConfig(
        enable_comprehensive_toolkit=True,
        enable_mcp_servers=True,
        tool_categories_enabled=[
            "analysis", "data_processing", "web_operations", 
            "file_operations", "research", "information_gathering"
        ],
        safety_level="standard",
        max_tools_per_category=30
    )


def create_creative_focused_config() -> ToolManagerConfig:
    """Create configuration focused on creative tools"""
    return ToolManagerConfig(
        enable_comprehensive_toolkit=True,
        enable_mcp_servers=True,
        tool_categories_enabled=[
            "creative_tools", "communication", "file_operations",
            "text_processing", "image_processing", "content_generation"
        ],
        safety_level="permissive",
        max_tools_per_category=40
    )


def create_secure_config() -> ToolManagerConfig:
    """Create a security-focused configuration"""
    return ToolManagerConfig(
        enable_comprehensive_toolkit=True,
        enable_mcp_servers=False,  # Disable MCP for security
        tool_categories_enabled=[
            "analysis", "security", "file_operations", "mathematics"
        ],
        safety_level="strict",
        max_tools_per_category=20,
        auto_discover_tools=False  # No auto-discovery for security
    )


if __name__ == "__main__":
    # Demo the tool manager
    import time
    
    print("Comprehensive Tool Manager Demo")
    print("=" * 40)
    
    # Create and initialize tool manager
    config = create_default_tool_config()
    manager = ToolManager(config)
    
    print("Initializing tool manager...")
    success = manager.initialize()
    
    if success:
        print("Tool manager initialized successfully!")
        
        # Show inventory
        inventory = manager.get_tool_inventory()
        print(f"\nTool Inventory:")
        print(f"  Classic Tools: {inventory['classic_tools']['count']}")
        print(f"  MCP Tools: {inventory['mcp_tools']['count']}")
        print(f"  Custom Tools: {inventory['custom_tools']['count']}")
        print(f"  Total Tools: {inventory['total_tools']}")
        
        # Show categories
        print(f"\nAvailable Categories:")
        for category in manager.get_available_categories():
            tool_count = len(manager.get_tools_by_category(category))
            print(f"  - {category}: {tool_count} tools")
        
        # Search demo
        print(f"\nSearch Results for 'file':")
        results = manager.search_tools("file")
        for result in results[:5]:  # Show first 5
            print(f"  - {result['name']} ({result['type']}): {result['description'][:50]}...")
        
        # Export configuration
        export_path = "tool_config_export.json"
        manager.export_tool_configuration(export_path)
        print(f"\nConfiguration exported to: {export_path}")
    
    else:
        print("Failed to initialize tool manager")
    
    # Cleanup
    manager.shutdown()