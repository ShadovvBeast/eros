"""
Main entry point for the Autonomous Logos-Pathos-Memory Agent.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from .config import AgentConfig
from .agent import AutonomousAgent
from .logging_config import setup_logging


def main():
    """Main entry point for the agent"""
    parser = argparse.ArgumentParser(description='Autonomous Logos-Pathos-Memory Agent')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--cycles', type=int, default=-1, help='Number of cycles to run (-1 for infinite)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-file', type=str, help='Path to log file')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring and instrumentation')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        enable_monitoring=not args.no_monitoring
    )
    
    try:
        # Create configuration
        config = AgentConfig.from_env()
        config.max_cycles = args.cycles
        config.log_level = args.log_level
        config.enable_monitoring = not args.no_monitoring
        
        # Create agent
        agent = AutonomousAgent(config)
        
        print(f"Autonomous Logos-Pathos-Memory Agent")
        print(f"Configuration: {config.pathos.state_dimension}D affective state")
        print(f"Cycles: {'infinite' if args.cycles < 0 else args.cycles}")
        print(f"Monitoring: {'enabled' if config.enable_monitoring else 'disabled'}")
        
        # Initialize all layers
        from .logos.logos_layer import LogosLayer
        from .pathos.pathos_layer import PathosLayer
        from .memory.memory_system import ConcreteMemorySystem
        from .ethos.ethos_framework import ConcreteEthosFramework
        from .tools.tool_layer import ToolLayer
        
        print("Initializing agent layers...")
        
        # Create layer instances
        ethos = ConcreteEthosFramework(config.ethos)
        logos = LogosLayer(config.logos)
        pathos = PathosLayer(config.pathos)
        memory = ConcreteMemorySystem(config.memory)
        tools = ToolLayer(config.tools, ethos)
        
        # Initialize agent with all layers
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        print("All layers initialized successfully!")
        print(f"Starting autonomous operation...")
        
        # Run autonomous operation
        agent.run_autonomous(max_cycles=args.cycles)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAgent stopped by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())