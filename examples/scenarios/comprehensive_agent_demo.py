"""
Comprehensive Agent Demo with Full Tool Suite

This scenario demonstrates the autonomous agent with the complete tool ecosystem,
including classic tools, MCP servers, and custom tools working together.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "tools"))

from src.agent import AutonomousAgent
from src.logos.logos_layer import LogosLayer
from src.pathos.pathos_layer import PathosLayer
from src.memory.memory_system import ConcreteMemorySystem
from src.ethos.ethos_framework import ConcreteEthosFramework
from src.tools.tool_layer import ToolLayer

# Import configurations
sys.path.append(str(Path(__file__).parent.parent / "configurations"))
from curious_explorer import create_curious_explorer_config

# Import tool systems
from easy_setup import EasyToolSetup
from tool_manager import ToolManager
from comprehensive_toolkit import get_comprehensive_toolkit
from mcp_integration import MCPIntegration


class ComprehensiveAgentDemo:
    """Demonstration of agent with comprehensive tool suite"""
    
    def __init__(self):
        self.agent = None
        self.tool_setup = None
        self.demo_tasks = [
            "analyze_system_performance",
            "create_content",
            "process_data",
            "web_research",
            "file_management",
            "security_operations",
            "mathematical_computation"
        ]
        self.current_task_index = 0
    
    def setup_comprehensive_agent(self, agent_type: str = "general") -> bool:
        """Set up agent with comprehensive tool suite"""
        try:
            print("🚀 Setting up Comprehensive Autonomous Agent")
            print("=" * 50)
            
            # Step 1: Easy tool setup
            print("📦 Step 1: Setting up comprehensive tool suite...")
            self.tool_setup = EasyToolSetup()
            success = self.tool_setup.quick_setup(
                agent_type=agent_type,
                enable_mcp=True,
                enable_web=True,
                enable_files=True,
                safety_level="standard"
            )
            
            if not success:
                print("❌ Failed to setup tools")
                return False
            
            # Step 2: Create agent configuration
            print("\n⚙️ Step 2: Configuring agent personality...")
            config = create_curious_explorer_config()
            
            # Customize for comprehensive demo
            config.ethos.identity_description = """
            I am a comprehensive autonomous agent equipped with a vast array of tools
            and capabilities. I can perform file operations, web research, data analysis,
            content creation, system monitoring, security operations, and mathematical
            computations. My goal is to demonstrate the full potential of an autonomous
            agent with access to comprehensive tooling.
            """
            
            config.ethos.core_values.extend([
                "tool_mastery",
                "comprehensive_analysis",
                "efficient_automation",
                "creative_problem_solving"
            ])
            
            # Step 3: Create and initialize agent
            print("\n🤖 Step 3: Initializing autonomous agent...")
            self.agent = AutonomousAgent(config)
            
            # Create layers
            ethos = ConcreteEthosFramework(config.ethos)
            logos = LogosLayer(config.logos)
            pathos = PathosLayer(config.pathos)
            memory = ConcreteMemorySystem(config.memory)
            
            # Use the comprehensive tool layer
            tools = self.tool_setup.get_tool_layer()
            
            # Initialize agent
            self.agent.initialize_layers(logos, pathos, memory, ethos, tools)
            
            print("✅ Comprehensive agent setup complete!")
            
            # Step 4: Show capabilities summary
            self._show_capabilities_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def _show_capabilities_summary(self) -> None:
        """Show a summary of agent capabilities"""
        if not self.tool_setup:
            return
        
        print("\n🎯 Agent Capabilities Summary:")
        print("-" * 30)
        
        inventory = self.tool_setup.get_tool_manager().get_tool_inventory()
        
        # Show tool counts
        print(f"📊 Tool Inventory:")
        print(f"   • Classic Tools: {inventory['classic_tools']['count']}")
        print(f"   • MCP Tools: {inventory['mcp_tools']['count']}")
        print(f"   • Custom Tools: {inventory['custom_tools']['count']}")
        print(f"   • Total Available: {inventory['total_tools']}")
        
        # Show key capabilities by category
        print(f"\n🔧 Key Capabilities:")
        
        capability_map = {
            "file_operations": "📁 File Management & Processing",
            "web_operations": "🌐 Web Research & Scraping", 
            "data_processing": "📈 Data Analysis & Processing",
            "creative_tools": "🎨 Content Creation & Design",
            "security": "🔒 Security & Encryption",
            "mathematics": "🧮 Mathematical Computation",
            "system_monitoring": "⚡ System Monitoring & Diagnostics",
            "communication": "💬 Communication & Collaboration"
        }
        
        for category, description in capability_map.items():
            tool_count = len(inventory['categories'].get(category, []))
            if tool_count > 0:
                print(f"   • {description}: {tool_count} tools")
        
        # Show MCP server status
        if inventory['mcp_tools']['count'] > 0:
            print(f"\n🔗 MCP Server Status:")
            servers = inventory['mcp_tools']['servers']['server_status']
            for server_name, status in servers.items():
                status_icon = "🟢" if status['running'] else "🔴"
                print(f"   {status_icon} {server_name}: {status['tool_count']} tools")
    
    def demonstrate_capabilities(self) -> None:
        """Demonstrate various agent capabilities"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
        
        print("\n🎪 Capability Demonstration")
        print("=" * 30)
        
        demonstrations = [
            ("File Operations", self._demo_file_operations),
            ("Data Analysis", self._demo_data_analysis),
            ("Web Operations", self._demo_web_operations),
            ("Creative Tools", self._demo_creative_tools),
            ("Security Operations", self._demo_security_operations),
            ("Mathematical Computation", self._demo_math_operations),
            ("System Monitoring", self._demo_system_monitoring)
        ]
        
        for demo_name, demo_func in demonstrations:
            print(f"\n🔍 Demonstrating: {demo_name}")
            print("-" * 25)
            
            try:
                demo_func()
                print(f"✅ {demo_name} demonstration complete")
            except Exception as e:
                print(f"❌ {demo_name} demonstration failed: {e}")
            
            time.sleep(1)  # Brief pause between demos
    
    def _demo_file_operations(self) -> None:
        """Demonstrate file operation capabilities"""
        tools = self.agent.tools
        
        # Create a test file
        test_content = "This is a test file created by the autonomous agent.\nIt demonstrates file creation capabilities."
        
        result = tools.execute_tool_by_name("write_file", {
            "path": "demo_output.txt",
            "content": test_content
        })
        
        if result.success:
            print("   📝 Created test file successfully")
            
            # Read it back
            read_result = tools.execute_tool_by_name("read_file", {
                "path": "demo_output.txt"
            })
            
            if read_result.success:
                print(f"   📖 Read file: {len(read_result.result['content'])} characters")
            
            # List directory
            list_result = tools.execute_tool_by_name("list_directory", {
                "path": "."
            })
            
            if list_result.success:
                file_count = list_result.result['total_files']
                print(f"   📂 Directory contains {file_count} files")
        else:
            print("   ❌ File operations not available")
    
    def _demo_data_analysis(self) -> None:
        """Demonstrate data analysis capabilities"""
        tools = self.agent.tools
        
        # Test statistical analysis
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
        
        result = tools.execute_tool_by_name("statistics_calculator", {
            "data": test_data,
            "operations": ["mean", "median", "std", "range"]
        })
        
        if result.success:
            stats = result.result['statistics']
            print(f"   📊 Mean: {stats.get('mean', 0):.2f}")
            print(f"   📊 Median: {stats.get('median', 0):.2f}")
            print(f"   📊 Std Dev: {stats.get('standard_deviation', 0):.2f}")
        
        # Test JSON processing
        json_data = {"name": "test", "values": [1, 2, 3], "active": True}
        
        json_result = tools.execute_tool_by_name("json_processor", {
            "operation": "stringify",
            "data": json_data,
            "indent": 2
        })
        
        if json_result.success:
            print("   🔄 JSON processing successful")
        else:
            print("   ❌ Data analysis tools not available")
    
    def _demo_web_operations(self) -> None:
        """Demonstrate web operation capabilities"""
        tools = self.agent.tools
        
        # Test HTTP request (to a safe endpoint)
        result = tools.execute_tool_by_name("http_request", {
            "url": "https://httpbin.org/json",
            "method": "GET"
        })
        
        if result.success:
            print("   🌐 HTTP request successful")
            print(f"   📡 Response status: {result.result.get('status_code', 'Unknown')}")
        else:
            print("   ❌ Web operations not available or network restricted")
    
    def _demo_creative_tools(self) -> None:
        """Demonstrate creative tool capabilities"""
        tools = self.agent.tools
        
        # Test text generation
        result = tools.execute_tool_by_name("text_generator", {
            "type": "story",
            "prompt": "artificial intelligence and creativity",
            "length": "short"
        })
        
        if result.success:
            generated_text = result.result['generated_text']
            print(f"   ✍️ Generated story: {len(generated_text)} characters")
            print(f"   📝 Word count: {result.result['word_count']}")
        
        # Test message formatting
        format_result = tools.execute_tool_by_name("message_formatter", {
            "message": "Hello from the autonomous agent!",
            "format": "markdown"
        })
        
        if format_result.success:
            print("   💬 Message formatting successful")
        else:
            print("   ❌ Creative tools not available")
    
    def _demo_security_operations(self) -> None:
        """Demonstrate security operation capabilities"""
        tools = self.agent.tools
        
        # Test hash generation
        result = tools.execute_tool_by_name("hash_generator", {
            "data": "autonomous agent security test",
            "algorithm": "sha256"
        })
        
        if result.success:
            hash_value = result.result['hash']
            print(f"   🔐 Generated SHA256 hash: {hash_value[:16]}...")
        
        # Test password generation
        pwd_result = tools.execute_tool_by_name("password_generator", {
            "length": 16,
            "include_symbols": True
        })
        
        if pwd_result.success:
            strength = pwd_result.result['strength']
            print(f"   🔑 Generated password strength: {strength}")
        
        # Test base64 encoding
        b64_result = tools.execute_tool_by_name("base64_processor", {
            "operation": "encode",
            "data": "autonomous agent"
        })
        
        if b64_result.success:
            print("   🔒 Base64 encoding successful")
        else:
            print("   ❌ Security tools not available")
    
    def _demo_math_operations(self) -> None:
        """Demonstrate mathematical computation capabilities"""
        tools = self.agent.tools
        
        # Test mathematical expression evaluation
        result = tools.execute_tool_by_name("math_evaluator", {
            "expression": "sqrt(16) + sin(pi/2) * 10",
            "variables": {"pi": 3.14159}
        })
        
        if result.success:
            math_result = result.result['result']
            print(f"   🧮 Math expression result: {math_result}")
        else:
            print("   ❌ Mathematical tools not available")
    
    def _demo_system_monitoring(self) -> None:
        """Demonstrate system monitoring capabilities"""
        tools = self.agent.tools
        
        # Test system information
        result = tools.execute_tool_by_name("system_info", {
            "type": "general"
        })
        
        if result.success:
            platform = result.result.get('platform', 'Unknown')
            print(f"   💻 System platform: {platform}")
            print(f"   🆔 Process ID: {result.result.get('process_id', 'Unknown')}")
        else:
            print("   ❌ System monitoring tools not available")
    
    def run_autonomous_session(self, cycles: int = 8) -> None:
        """Run an autonomous session demonstrating tool usage"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
        
        print(f"\n🤖 Running Autonomous Session ({cycles} cycles)")
        print("=" * 40)
        
        # Track tool usage
        tool_usage_stats = {}
        session_metrics = {
            "total_tools_used": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "unique_tools_used": set()
        }
        
        for cycle in range(cycles):
            print(f"\n🔄 Cycle {cycle + 1}/{cycles}")
            print("-" * 20)
            
            try:
                result = self.agent.run_cycle()
                
                # Track metrics
                tool_used = result.get('tool_used')
                if tool_used:
                    session_metrics["total_tools_used"] += 1
                    session_metrics["unique_tools_used"].add(tool_used)
                    
                    if tool_used not in tool_usage_stats:
                        tool_usage_stats[tool_used] = 0
                    tool_usage_stats[tool_used] += 1
                    
                    if result.get('external_reward', 0) >= 0:
                        session_metrics["successful_operations"] += 1
                    else:
                        session_metrics["failed_operations"] += 1
                
                # Display cycle results
                print(f"   🎯 Intention: {result['intention'][:60]}...")
                print(f"   🔧 Tool Used: {tool_used or 'None'}")
                print(f"   💝 Internal Reward: {result['internal_reward']:.3f}")
                print(f"   🌟 External Reward: {result['external_reward']:.3f}")
                print(f"   💾 Memory Stored: {result['memory_stored']}")
                
                # Brief pause
                time.sleep(0.2)
                
            except Exception as e:
                print(f"   ❌ Cycle {cycle + 1} failed: {e}")
                session_metrics["failed_operations"] += 1
        
        # Display session summary
        self._display_session_summary(session_metrics, tool_usage_stats)
    
    def _display_session_summary(self, metrics: Dict[str, Any], tool_stats: Dict[str, int]) -> None:
        """Display autonomous session summary"""
        print(f"\n📈 Autonomous Session Summary")
        print("=" * 30)
        
        print(f"🔧 Tool Usage:")
        print(f"   • Total tool calls: {metrics['total_tools_used']}")
        print(f"   • Unique tools used: {len(metrics['unique_tools_used'])}")
        print(f"   • Successful operations: {metrics['successful_operations']}")
        print(f"   • Failed operations: {metrics['failed_operations']}")
        
        if tool_stats:
            print(f"\n🏆 Most Used Tools:")
            sorted_tools = sorted(tool_stats.items(), key=lambda x: x[1], reverse=True)
            for tool_name, usage_count in sorted_tools[:5]:
                print(f"   • {tool_name}: {usage_count} times")
        
        # Agent status
        if self.agent:
            status = self.agent.get_status()
            print(f"\n🤖 Agent Status:")
            print(f"   • Total cycles: {status['cycle_count']}")
            print(f"   • Memory traces: {status['memory_trace_count']}")
            print(f"   • Current state norm: {status['current_state_norm']:.3f}")
    
    def interactive_tool_explorer(self) -> None:
        """Interactive tool exploration interface"""
        if not self.tool_setup:
            print("❌ Tool setup not initialized")
            return
        
        print("\n🔍 Interactive Tool Explorer")
        print("=" * 30)
        
        while True:
            print("\nOptions:")
            print("1. List all tools")
            print("2. Search tools")
            print("3. Test a specific tool")
            print("4. Show tool categories")
            print("5. Show MCP server status")
            print("6. Exit explorer")
            
            try:
                choice = input("\nEnter choice (1-6): ").strip()
                
                if choice == "1":
                    self.tool_setup.list_available_tools()
                
                elif choice == "2":
                    query = input("Enter search query: ").strip()
                    self.tool_setup.search_tools(query)
                
                elif choice == "3":
                    self._interactive_tool_test()
                
                elif choice == "4":
                    categories = self.tool_setup.get_tool_manager().get_available_categories()
                    print(f"\nAvailable categories:")
                    for i, category in enumerate(categories, 1):
                        tool_count = len(self.tool_setup.get_tool_manager().get_tools_by_category(category))
                        print(f"   {i}. {category} ({tool_count} tools)")
                
                elif choice == "5":
                    if self.tool_setup.get_tool_manager().mcp_integration:
                        status = self.tool_setup.get_tool_manager().mcp_integration.get_server_status()
                        print(f"\nMCP Server Status:")
                        for server, info in status['server_status'].items():
                            status_icon = "🟢" if info['running'] else "🔴"
                            print(f"   {status_icon} {server}: {info['tool_count']} tools")
                    else:
                        print("❌ MCP integration not available")
                
                elif choice == "6":
                    break
                
                else:
                    print("❌ Invalid choice")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting tool explorer")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _interactive_tool_test(self) -> None:
        """Interactive tool testing"""
        tool_name = input("Enter tool name to test: ").strip()
        
        if not self.agent or not self.agent.tools:
            print("❌ Agent tools not available")
            return
        
        available_tools = self.agent.tools.get_available_tools()
        
        if tool_name not in available_tools:
            print(f"❌ Tool '{tool_name}' not found")
            print(f"Available tools: {', '.join(available_tools[:10])}...")
            return
        
        print(f"\n🧪 Testing tool: {tool_name}")
        
        # Get tool description
        description = self.agent.tools.get_tool_description(tool_name)
        if description:
            print(f"Description: {description}")
        
        # Simple test with empty arguments
        try:
            result = self.agent.tools.execute_tool_by_name(tool_name, {})
            
            if result.success:
                print("✅ Tool test successful!")
                print(f"Result: {str(result.result)[:200]}...")
            else:
                print(f"❌ Tool test failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Tool test error: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the comprehensive agent demo"""
        if self.tool_setup:
            self.tool_setup.shutdown()
        
        print("🛑 Comprehensive agent demo shutdown complete")


def main():
    """Main function for comprehensive agent demo"""
    demo = ComprehensiveAgentDemo()
    
    try:
        print("🎯 Comprehensive Autonomous Agent Demo")
        print("=" * 45)
        
        # Setup agent
        success = demo.setup_comprehensive_agent("general")
        
        if not success:
            print("❌ Failed to setup agent")
            return
        
        # Interactive menu
        while True:
            print("\n🎮 Demo Options:")
            print("1. Demonstrate capabilities")
            print("2. Run autonomous session")
            print("3. Interactive tool explorer")
            print("4. Show agent status")
            print("5. Exit demo")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                demo.demonstrate_capabilities()
            
            elif choice == "2":
                cycles = input("Enter number of cycles (default 8): ").strip()
                cycles = int(cycles) if cycles.isdigit() else 8
                demo.run_autonomous_session(cycles)
            
            elif choice == "3":
                demo.interactive_tool_explorer()
            
            elif choice == "4":
                if demo.agent:
                    status = demo.agent.get_status()
                    print(f"\n🤖 Agent Status:")
                    for key, value in status.items():
                        print(f"   • {key}: {value}")
                else:
                    print("❌ Agent not initialized")
            
            elif choice == "5":
                break
            
            else:
                print("❌ Invalid choice")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main()