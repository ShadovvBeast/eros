"""
Easy Setup for Comprehensive Agent Tools

This module provides simple, one-line setup for the complete tool ecosystem,
including both classic tools and MCP servers.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tool_manager import ToolManager, ToolManagerConfig, create_default_tool_config
from mcp_integration import MCPServerConfig, create_mcp_config_file, get_default_mcp_servers
from src.tools.tool_layer import ToolLayer
from src.config import ToolConfig
from src.ethos.interfaces import EthosFramework

logger = logging.getLogger(__name__)


class EasyToolSetup:
    """Easy setup class for comprehensive agent tools"""
    
    def __init__(self):
        self.tool_manager: Optional[ToolManager] = None
        self.tool_layer: Optional[ToolLayer] = None
        self.setup_complete = False
        
    def quick_setup(self, 
                   agent_type: str = "general",
                   enable_mcp: bool = True,
                   enable_web: bool = True,
                   enable_files: bool = True,
                   safety_level: str = "standard") -> bool:
        """
        Quick one-line setup for agent tools
        
        Args:
            agent_type: Type of agent (general, research, creative, secure)
            enable_mcp: Enable MCP server integration
            enable_web: Enable web-related tools
            enable_files: Enable file system tools
            safety_level: Safety level (strict, standard, permissive)
        
        Returns:
            True if setup successful
        """
        try:
            print(f"🚀 Setting up {agent_type} agent with comprehensive tools...")
            
            # Create appropriate configuration
            config = self._create_config_for_agent_type(
                agent_type, enable_mcp, enable_web, enable_files, safety_level
            )
            
            # Initialize tool manager
            self.tool_manager = ToolManager(config)
            success = self.tool_manager.initialize()
            
            if not success:
                print("❌ Failed to initialize tool manager")
                return False
            
            # Create and setup tool layer
            tool_config = ToolConfig(
                max_concurrent_tools=5,
                tool_timeout_seconds=30,
                allowed_tool_types=self._get_allowed_tool_types(agent_type)
            )
            
            self.tool_layer = ToolLayer(tool_config)
            
            # Register all tools
            registered_count = self.tool_manager.register_with_tool_layer(self.tool_layer)
            
            print(f"✅ Setup complete! Registered {registered_count} tools")
            
            # Show summary
            self._show_setup_summary()
            
            self.setup_complete = True
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            logger.error(f"Quick setup failed: {e}")
            return False
    
    def _create_config_for_agent_type(self, 
                                    agent_type: str,
                                    enable_mcp: bool,
                                    enable_web: bool, 
                                    enable_files: bool,
                                    safety_level: str) -> ToolManagerConfig:
        """Create configuration based on agent type"""
        
        base_config = create_default_tool_config()
        base_config.enable_mcp_servers = enable_mcp
        base_config.safety_level = safety_level
        
        # Configure categories based on agent type and preferences
        if agent_type == "research":
            categories = ["analysis", "data_processing", "research", "information_gathering"]
            if enable_web:
                categories.extend(["web_operations"])
            if enable_files:
                categories.extend(["file_operations"])
            base_config.tool_categories_enabled = categories
            
        elif agent_type == "creative":
            categories = ["creative_tools", "communication", "content_generation"]
            if enable_files:
                categories.extend(["file_operations"])
            if enable_web:
                categories.extend(["web_operations"])
            base_config.tool_categories_enabled = categories
            base_config.safety_level = "permissive"  # More permissive for creativity
            
        elif agent_type == "secure":
            categories = ["analysis", "security", "mathematics"]
            if enable_files:
                categories.extend(["file_operations"])
            base_config.tool_categories_enabled = categories
            base_config.enable_mcp_servers = False  # Disable MCP for security
            base_config.safety_level = "strict"
            base_config.auto_discover_tools = False
            
        elif agent_type == "developer":
            categories = ["development", "file_operations", "system_operations", "analysis"]
            if enable_web:
                categories.extend(["web_operations"])
            base_config.tool_categories_enabled = categories
            
        # For "general" type, use all categories (None means all enabled)
        
        return base_config
    
    def _get_allowed_tool_types(self, agent_type: str) -> List[str]:
        """Get allowed tool types for ToolLayer based on agent type"""
        base_types = [
            "analysis", "communication", "information_gathering", 
            "learning", "mcp_tools"
        ]
        
        if agent_type == "research":
            return base_types + [
                "web_search", "data_analysis", "research", "file_operations"
            ]
        elif agent_type == "creative":
            return base_types + [
                "creative_tools", "design_software", "artistic_creation", 
                "file_operations"
            ]
        elif agent_type == "secure":
            return [
                "analysis", "security", "mathematics", "file_operations"
            ]
        elif agent_type == "developer":
            return base_types + [
                "development", "file_operations", "system_operations",
                "web_operations"
            ]
        else:  # general
            return [
                "web_search", "file_operations", "analysis", "communication",
                "learning", "creative_tools", "mcp_tools", "information_gathering",
                "data_processing", "system_monitoring", "security", "mathematics"
            ]
    
    def _show_setup_summary(self) -> None:
        """Show a summary of the setup"""
        if not self.tool_manager:
            return
        
        inventory = self.tool_manager.get_tool_inventory()
        
        print("\n📊 Setup Summary:")
        print(f"   Classic Tools: {inventory['classic_tools']['count']}")
        print(f"   MCP Tools: {inventory['mcp_tools']['count']}")
        print(f"   Custom Tools: {inventory['custom_tools']['count']}")
        print(f"   Total Tools: {inventory['total_tools']}")
        
        print("\n📂 Tool Categories:")
        for category, tools in inventory['categories'].items():
            print(f"   - {category}: {len(tools)} tools")
        
        # Show MCP server status if available
        if inventory['mcp_tools']['count'] > 0:
            print("\n🔗 MCP Server Status:")
            servers = inventory['mcp_tools']['servers']['server_status']
            for server_name, status in servers.items():
                status_icon = "🟢" if status['running'] else "🔴"
                print(f"   {status_icon} {server_name}: {status['tool_count']} tools")
    
    def get_tool_layer(self) -> Optional[ToolLayer]:
        """Get the configured tool layer"""
        return self.tool_layer
    
    def get_tool_manager(self) -> Optional[ToolManager]:
        """Get the tool manager"""
        return self.tool_manager
    
    def list_available_tools(self, category: Optional[str] = None) -> None:
        """List all available tools, optionally filtered by category"""
        if not self.tool_manager:
            print("❌ Tool manager not initialized")
            return
        
        inventory = self.tool_manager.get_tool_inventory()
        
        if category:
            tools = self.tool_manager.get_tools_by_category(category)
            print(f"\n🔧 Tools in category '{category}':")
            
            # Show tools from each type
            for tool_name in tools:
                # Find tool description
                description = "No description"
                
                if tool_name in inventory['classic_tools']['tools']:
                    description = inventory['classic_tools']['tools'][tool_name]['description']
                elif tool_name in inventory['mcp_tools']['tools']:
                    description = inventory['mcp_tools']['tools'][tool_name]['description']
                elif tool_name in inventory['custom_tools']['tools']:
                    description = inventory['custom_tools']['tools'][tool_name]['description']
                
                print(f"   - {tool_name}: {description}")
        else:
            print("\n🔧 All Available Tools:")
            
            # Show by category
            for cat, tools in inventory['categories'].items():
                print(f"\n   📁 {cat.upper()}:")
                for tool_name in tools[:5]:  # Show first 5 per category
                    description = "No description"
                    
                    if tool_name in inventory['classic_tools']['tools']:
                        description = inventory['classic_tools']['tools'][tool_name]['description']
                    elif tool_name in inventory['mcp_tools']['tools']:
                        description = inventory['mcp_tools']['tools'][tool_name]['description']
                    elif tool_name in inventory['custom_tools']['tools']:
                        description = inventory['custom_tools']['tools'][tool_name]['description']
                    
                    print(f"     - {tool_name}: {description[:60]}...")
                
                if len(tools) > 5:
                    print(f"     ... and {len(tools) - 5} more tools")
    
    def search_tools(self, query: str) -> None:
        """Search for tools by name or description"""
        if not self.tool_manager:
            print("❌ Tool manager not initialized")
            return
        
        results = self.tool_manager.search_tools(query)
        
        if results:
            print(f"\n🔍 Search results for '{query}':")
            for result in results:
                type_icon = {"classic": "🔧", "mcp": "🔗", "custom": "⚙️"}
                icon = type_icon.get(result['type'], "❓")
                print(f"   {icon} {result['name']}: {result['description']}")
        else:
            print(f"❌ No tools found matching '{query}'")
    
    def create_mcp_config(self, config_path: str = ".kiro/settings/mcp.json") -> bool:
        """Create a default MCP configuration file"""
        try:
            servers = get_default_mcp_servers()
            success = create_mcp_config_file(servers, config_path)
            
            if success:
                print(f"✅ Created MCP configuration: {config_path}")
                print("   You can edit this file to customize MCP servers")
                return True
            else:
                print(f"❌ Failed to create MCP configuration")
                return False
                
        except Exception as e:
            print(f"❌ Error creating MCP config: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown and cleanup"""
        if self.tool_manager:
            self.tool_manager.shutdown()
        print("🛑 Tool system shutdown complete")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_research_agent() -> EasyToolSetup:
    """One-line setup for a research-focused agent"""
    setup = EasyToolSetup()
    setup.quick_setup(
        agent_type="research",
        enable_mcp=True,
        enable_web=True,
        enable_files=True,
        safety_level="standard"
    )
    return setup


def setup_creative_agent() -> EasyToolSetup:
    """One-line setup for a creative-focused agent"""
    setup = EasyToolSetup()
    setup.quick_setup(
        agent_type="creative",
        enable_mcp=True,
        enable_web=True,
        enable_files=True,
        safety_level="permissive"
    )
    return setup


def setup_secure_agent() -> EasyToolSetup:
    """One-line setup for a security-focused agent"""
    setup = EasyToolSetup()
    setup.quick_setup(
        agent_type="secure",
        enable_mcp=False,
        enable_web=False,
        enable_files=True,
        safety_level="strict"
    )
    return setup


def setup_developer_agent() -> EasyToolSetup:
    """One-line setup for a development-focused agent"""
    setup = EasyToolSetup()
    setup.quick_setup(
        agent_type="developer",
        enable_mcp=True,
        enable_web=True,
        enable_files=True,
        safety_level="standard"
    )
    return setup


def setup_general_agent() -> EasyToolSetup:
    """One-line setup for a general-purpose agent"""
    setup = EasyToolSetup()
    setup.quick_setup(
        agent_type="general",
        enable_mcp=True,
        enable_web=True,
        enable_files=True,
        safety_level="standard"
    )
    return setup


# ============================================================================
# EXAMPLE USAGE AND DEMO
# ============================================================================

def demo_easy_setup():
    """Demonstrate the easy setup system"""
    print("🎯 Easy Tool Setup Demo")
    print("=" * 30)
    
    # Demo different agent types
    agent_types = [
        ("research", "Research Agent"),
        ("creative", "Creative Agent"), 
        ("secure", "Secure Agent"),
        ("developer", "Developer Agent"),
        ("general", "General Agent")
    ]
    
    for agent_type, display_name in agent_types:
        print(f"\n🤖 Setting up {display_name}...")
        
        setup = EasyToolSetup()
        success = setup.quick_setup(agent_type=agent_type)
        
        if success:
            print(f"✅ {display_name} setup successful!")
            
            # Show a few tools
            inventory = setup.get_tool_manager().get_tool_inventory()
            print(f"   Available tools: {inventory['total_tools']}")
            
            # Show categories
            categories = list(inventory['categories'].keys())[:3]
            print(f"   Sample categories: {', '.join(categories)}")
        else:
            print(f"❌ {display_name} setup failed!")
        
        # Cleanup
        setup.shutdown()
        print()


def interactive_demo():
    """Interactive demo of the tool system"""
    print("🎮 Interactive Tool System Demo")
    print("=" * 35)
    
    # Let user choose agent type
    print("\nChoose an agent type:")
    print("1. Research Agent (analysis, web search, data processing)")
    print("2. Creative Agent (content generation, design tools)")
    print("3. Secure Agent (limited, security-focused tools)")
    print("4. Developer Agent (development, system tools)")
    print("5. General Agent (all tools available)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        agent_map = {
            "1": "research",
            "2": "creative", 
            "3": "secure",
            "4": "developer",
            "5": "general"
        }
        
        agent_type = agent_map.get(choice, "general")
        
        print(f"\n🚀 Setting up {agent_type} agent...")
        
        setup = EasyToolSetup()
        success = setup.quick_setup(agent_type=agent_type)
        
        if success:
            print("\n✅ Setup complete! What would you like to do?")
            
            while True:
                print("\nOptions:")
                print("1. List all tools")
                print("2. List tools by category")
                print("3. Search tools")
                print("4. Show setup summary")
                print("5. Exit")
                
                action = input("\nChoose action (1-5): ").strip()
                
                if action == "1":
                    setup.list_available_tools()
                elif action == "2":
                    categories = setup.get_tool_manager().get_available_categories()
                    print(f"\nAvailable categories: {', '.join(categories)}")
                    category = input("Enter category name: ").strip()
                    if category in categories:
                        setup.list_available_tools(category)
                    else:
                        print("❌ Invalid category")
                elif action == "3":
                    query = input("Enter search query: ").strip()
                    setup.search_tools(query)
                elif action == "4":
                    setup._show_setup_summary()
                elif action == "5":
                    break
                else:
                    print("❌ Invalid choice")
            
            setup.shutdown()
        else:
            print("❌ Setup failed!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        demo_easy_setup()