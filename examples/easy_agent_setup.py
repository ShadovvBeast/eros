"""
Easy Agent Setup with Comprehensive Tools

This module provides simple, one-line setup for agents with the complete tool ecosystem.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agent import AutonomousAgent
from src.logos.logos_layer import LogosLayer
from src.pathos.pathos_layer import PathosLayer
from src.memory.memory_system import ConcreteMemorySystem
from src.ethos.ethos_framework import ConcreteEthosFramework
from src.tools import ToolManager, ToolLayer, create_default_config
from src.config import AgentConfig, ToolConfig

# Import configurations
sys.path.append(str(Path(__file__).parent / "configurations"))
from curious_explorer import create_curious_explorer_config
from cautious_analyst import create_cautious_analyst_config
from creative_collaborator import create_creative_collaborator_config


def setup_agent_with_comprehensive_tools(agent_type: str = "general", 
                                        enable_mcp: bool = True,
                                        safety_level: str = "standard") -> tuple:
    """
    One-line setup for an autonomous agent with comprehensive tools.
    
    Args:
        agent_type: Type of agent (general, research, creative, secure)
        enable_mcp: Enable MCP server integration
        safety_level: Safety level (strict, standard, permissive)
    
    Returns:
        Tuple of (agent, tool_manager) ready for use
    """
    print(f"🚀 Setting up {agent_type} agent with comprehensive tools...")
    
    # Step 1: Create tool manager configuration
    if agent_type == "research":
        from src.tools.manager import create_research_config
        tool_config = create_research_config()
    elif agent_type == "creative":
        from src.tools.manager import create_creative_config
        tool_config = create_creative_config()
    elif agent_type == "secure":
        from src.tools.manager import create_secure_config
        tool_config = create_secure_config()
    else:  # general
        tool_config = create_default_config()
    
    tool_config.enable_mcp_servers = enable_mcp
    tool_config.safety_level = safety_level
    
    # Step 2: Initialize tool manager
    tool_manager = ToolManager(tool_config)
    success = tool_manager.initialize()
    
    if not success:
        raise RuntimeError("Failed to initialize tool manager")
    
    # Step 3: Create agent configuration
    if agent_type == "research":
        agent_config = create_curious_explorer_config()  # Good for research
    elif agent_type == "creative":
        agent_config = create_creative_collaborator_config()
    elif agent_type == "secure":
        agent_config = create_cautious_analyst_config()  # Good for security
    else:  # general
        agent_config = create_curious_explorer_config()
    
    # Step 4: Create and initialize agent
    agent = AutonomousAgent(agent_config)
    
    # Create layers
    ethos = ConcreteEthosFramework(agent_config.ethos)
    logos = LogosLayer(agent_config.logos)
    pathos = PathosLayer(agent_config.pathos)
    memory = ConcreteMemorySystem(agent_config.memory)
    
    # Create tool layer and register all tools
    tool_layer_config = ToolConfig(
        max_concurrent_tools=5,
        tool_timeout_seconds=30,
        allowed_tool_types=[
            "file_operations", "web_operations", "data_processing",
            "security", "mathematics", "system_operations", "mcp_tools"
        ]
    )
    tool_layer = ToolLayer(tool_layer_config, ethos)
    
    # Register all tools from tool manager
    registered_count = tool_manager.register_with_tool_layer(tool_layer)
    
    # Initialize agent
    agent.initialize_layers(logos, pathos, memory, ethos, tool_layer)
    
    print(f"✅ Agent setup complete! Registered {registered_count} tools")
    
    # Show summary
    inventory = tool_manager.get_tool_inventory()
    print(f"📊 Tool Summary:")
    print(f"   • Built-in Tools: {inventory['builtin_tools']['count']}")
    print(f"   • MCP Tools: {inventory['mcp_tools']['count']}")
    print(f"   • Total Tools: {inventory['total_tools']}")
    print(f"   • Categories: {len(inventory['categories'])}")
    
    return agent, tool_manager


# Convenience functions for specific agent types
def setup_research_agent(enable_mcp: bool = True) -> tuple:
    """Setup a research-focused agent"""
    return setup_agent_with_comprehensive_tools("research", enable_mcp, "standard")


def setup_creative_agent(enable_mcp: bool = True) -> tuple:
    """Setup a creative-focused agent"""
    return setup_agent_with_comprehensive_tools("creative", enable_mcp, "permissive")


def setup_secure_agent(enable_mcp: bool = False) -> tuple:
    """Setup a security-focused agent"""
    return setup_agent_with_comprehensive_tools("secure", enable_mcp, "strict")


def setup_general_agent(enable_mcp: bool = True) -> tuple:
    """Setup a general-purpose agent"""
    return setup_agent_with_comprehensive_tools("general", enable_mcp, "standard")


if __name__ == "__main__":
    # Demo the easy setup
    print("🎯 Easy Agent Setup Demo")
    print("=" * 30)
    
    try:
        # Setup a general agent
        agent, tool_manager = setup_general_agent()
        
        print("\n🔧 Available Tool Categories:")
        for category in tool_manager.get_available_categories():
            tool_count = len(tool_manager.get_tools_by_category(category))
            print(f"   • {category}: {tool_count} tools")
        
        print("\n🧪 Testing a few tools...")
        
        # Test system info
        result = agent.tools.execute_tool_by_name("system_info", {"type": "general"})
        if result.success:
            print("   ✅ System info tool working")
        
        # Test math tool
        result = agent.tools.execute_tool_by_name("math_evaluator", {
            "expression": "2 + 2 * 3"
        })
        if result.success:
            print(f"   ✅ Math evaluator: 2 + 2 * 3 = {result.result['result']}")
        
        # Test JSON processor
        result = agent.tools.execute_tool_by_name("json_processor", {
            "operation": "stringify",
            "data": {"test": "data", "number": 42}
        })
        if result.success:
            print("   ✅ JSON processor working")
        
        print("\n🎉 Demo complete! Agent is ready for autonomous operation.")
        
        # Cleanup
        tool_manager.shutdown()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()