#!/usr/bin/env python3
"""
Simple Autonomous Agent Test

This script demonstrates a basic autonomous agent running a few cycles
to test the system functionality and monitor behavior.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent import AutonomousAgent
from src.logos.logos_layer import LogosLayer
from src.pathos.pathos_layer import PathosLayer
from src.memory.memory_system import ConcreteMemorySystem
from src.ethos.ethos_framework import ConcreteEthosFramework
from src.tools.tool_layer import ToolLayer
from src.config import AgentConfig, PathosConfig, LogosConfig, MemoryConfig, EthosConfig


def create_simple_test_config():
    """Create a simple configuration for testing"""
    
    # Ethos configuration - simple curious agent
    ethos_config = EthosConfig(
        identity_description="""
        I am a simple autonomous agent designed for testing and demonstration.
        I am curious about the world and enjoy exploring different tools and capabilities.
        My goal is to learn and demonstrate autonomous behavior while staying safe.
        """,
        core_values=[
            "curiosity",
            "learning", 
            "safety",
            "exploration",
            "helpfulness"
        ],
        safety_constraints=[
            "no_harmful_actions",
            "respect_privacy",
            "stay_within_bounds",
            "no_system_modification"
        ],
        allowed_tool_categories=[
            "analysis",
            "communication",
            "web_search",
            "file_operations"
        ],
        personality_traits={
            'openness': 0.8,
            'conscientiousness': 0.7,
            'extraversion': 0.6,
            'agreeableness': 0.8,
            'neuroticism': 0.3,
            'curiosity': 0.9,
            'creativity': 0.7,
            'caution': 0.6
        }
    )
    
    # Pathos configuration - moderate complexity
    pathos_config = PathosConfig(
        state_dimension=64,  # Smaller for testing
        decay_factor=0.95,
        echo_strength=0.1,
        homeostatic_weights={
            'stability': 1.0,
            'complexity': 0.5,
            'coherence': 0.8
        },
        salience_threshold=0.6,  # Lower threshold for more memories
        memory_retrieval_k=3
    )
    
    # Logos configuration
    logos_config = LogosConfig(
        semantic_dimension=256,  # Smaller for testing
        max_intentions_per_cycle=2,
        planning_horizon=3,
        interest_threshold=0.5,
        gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
        gemini_model='gemini-1.5-flash'
    )
    
    # Memory configuration
    memory_config = MemoryConfig(
        max_memory_traces=1000,
        similarity_threshold=0.7,
        novelty_k_neighbors=5,
        storage_backend='memory'  # Use in-memory for testing
    )
    
    config = AgentConfig(
        ethos=ethos_config,
        pathos=pathos_config,
        logos=logos_config,
        memory=memory_config
    )
    
    # Set additional parameters
    config.cycle_interval_seconds = 1.0
    config.enable_monitoring = True
    config.log_level = "INFO"
    
    return config


class SimpleMathTool:
    """Simple math operations tool"""
    
    @property
    def name(self) -> str:
        return "simple_math"
    
    @property
    def description(self) -> str:
        return "Perform simple mathematical operations (add, subtract, multiply, divide)"
    
    @property
    def category(self) -> str:
        return "analysis"
    
    def execute(self, arguments: Dict[str, Any]):
        """Execute math operation"""
        from src.models import ToolResult
        
        try:
            operation = arguments.get('operation', 'add')
            a = float(arguments.get('a', 0))
            b = float(arguments.get('b', 0))
            
            if operation == 'add':
                result = a + b
            elif operation == 'multiply':
                result = a * b
            elif operation == 'subtract':
                result = a - b
            elif operation == 'divide':
                result = a / b if b != 0 else 0
            else:
                result = 0
                
            return ToolResult(
                success=True,
                result={
                    'result': result,
                    'operation': operation,
                    'inputs': {'a': a, 'b': b}
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_arguments(self, arguments: Dict[str, Any]):
        """Validate arguments"""
        return True, None


class TextAnalysisTool:
    """Simple text analysis tool"""
    
    @property
    def name(self) -> str:
        return "text_analysis"
    
    @property
    def description(self) -> str:
        return "Analyze text for basic statistics (word count, character count, etc.)"
    
    @property
    def category(self) -> str:
        return "analysis"
    
    def execute(self, arguments: Dict[str, Any]):
        """Execute text analysis"""
        from src.models import ToolResult
        
        try:
            text = arguments.get('text', '')
            
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            
            return ToolResult(
                success=True,
                result={
                    'word_count': word_count,
                    'character_count': char_count,
                    'sentence_count': sentence_count,
                    'text_length': len(text)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_arguments(self, arguments: Dict[str, Any]):
        """Validate arguments"""
        return True, None


class RandomFactTool:
    """Random fact generator tool"""
    
    @property
    def name(self) -> str:
        return "random_fact"
    
    @property
    def description(self) -> str:
        return "Get a random interesting fact"
    
    @property
    def category(self) -> str:
        return "communication"  # Changed to match allowed categories
    
    def execute(self, arguments: Dict[str, Any]):
        """Execute random fact generation"""
        from src.models import ToolResult
        
        facts = [
            "Octopuses have three hearts and blue blood.",
            "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
            "A group of flamingos is called a 'flamboyance'.",
            "Bananas are berries, but strawberries aren't.",
            "The shortest war in history lasted only 38-45 minutes.",
            "Dolphins have names for each other.",
            "There are more possible games of chess than atoms in the observable universe.",
            "Wombat poop is cube-shaped.",
            "A single cloud can weigh more than a million pounds.",
            "Butterflies taste with their feet."
        ]
        
        import random
        fact = random.choice(facts)
        
        return ToolResult(
            success=True,
            result={
                'fact': fact,
                'category': 'random_knowledge'
            }
        )
    
    def validate_arguments(self, arguments: Dict[str, Any]):
        """Validate arguments"""
        return True, None


def create_simple_tools():
    """Create a simple set of safe tools for testing"""
    return [
        SimpleMathTool(),
        TextAnalysisTool(),
        RandomFactTool()
    ]


def run_simple_agent_test(cycles=5):
    """Run a simple autonomous agent test"""
    
    print("🤖 Simple Autonomous Agent Test")
    print("=" * 40)
    
    try:
        # Create configuration
        print("⚙️ Creating agent configuration...")
        config = create_simple_test_config()
        
        # Create agent
        print("🚀 Initializing autonomous agent...")
        agent = AutonomousAgent(config)
        
        # Create layers
        ethos = ConcreteEthosFramework(config.ethos)
        logos = LogosLayer(config.logos)
        pathos = PathosLayer(config.pathos)
        memory = ConcreteMemorySystem(config.memory)
        tools = ToolLayer(config.tools, ethos)
        
        # Register simple tools
        print("🔧 Registering tools...")
        simple_tools = create_simple_tools()
        for tool in simple_tools:
            try:
                tools.register_tool(tool)
            except Exception as e:
                print(f"   ⚠️ Failed to register {tool.name}: {e}")
        
        print(f"   Registered {len(simple_tools)} tools")
        
        # Initialize agent
        print("🎯 Initializing agent layers...")
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        print("✅ Agent initialization complete!")
        print(f"\n🔄 Running {cycles} autonomous cycles...")
        print("-" * 30)
        
        # Track metrics
        cycle_results = []
        
        # Run cycles
        for cycle in range(cycles):
            print(f"\n🔄 Cycle {cycle + 1}/{cycles}")
            
            try:
                # Run single cycle
                result = agent.run_cycle()
                cycle_results.append(result)
                
                # Display results
                print(f"   💭 Intention: {result['intention'][:80]}...")
                print(f"   🔧 Tool Used: {result.get('tool_used', 'None')}")
                print(f"   💝 Internal Reward: {result['internal_reward']:.3f}")
                print(f"   🌟 External Reward: {result['external_reward']:.3f}")
                print(f"   💾 Memory Stored: {result['memory_stored']}")
                
                # Brief pause
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Cycle {cycle + 1} failed: {e}")
                continue
        
        # Display summary
        print(f"\n📊 Test Summary")
        print("=" * 20)
        
        successful_cycles = len([r for r in cycle_results if r])
        tools_used = [r.get('tool_used') for r in cycle_results if r.get('tool_used')]
        memories_stored = sum(1 for r in cycle_results if r.get('memory_stored'))
        
        print(f"✅ Successful cycles: {successful_cycles}/{cycles}")
        print(f"🔧 Tools used: {len(tools_used)} times")
        print(f"💾 Memories stored: {memories_stored}")
        
        if tools_used:
            print(f"🏆 Most used tools: {', '.join(set(tools_used))}")
        
        # Agent status
        status = agent.get_status()
        print(f"\n🤖 Final Agent Status:")
        print(f"   • Total cycles: {status['cycle_count']}")
        print(f"   • Memory traces: {status['memory_trace_count']}")
        print(f"   • Current state norm: {status['current_state_norm']:.3f}")
        
        print(f"\n✅ Simple agent test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("🎯 Starting Simple Autonomous Agent Test")
    
    # Load environment variables from .env file
    from pathlib import Path
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Check environment
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️ Warning: GEMINI_API_KEY not found in environment")
        print("   The agent may not function properly without an API key")
    else:
        print("✅ GEMINI_API_KEY found in environment")
    
    # Run test
    cycles = 5
    print(f"🔄 Will run {cycles} autonomous cycles")
    
    success = run_simple_agent_test(cycles)
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
    
    return success


if __name__ == "__main__":
    main()