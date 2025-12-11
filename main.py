#!/usr/bin/env python3
"""
Main entry point for the Autonomous Logos-Pathos-Memory Agent system.

This script provides a unified interface to launch different components of the system.
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(
        description="E.R.O.S — Enhanced Reasoning & Operational System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  gui         - Launch the E.R.O.S Control Center (recommended)
  ultimate    - Run E.R.O.S with extended autonomous capabilities
  simple      - Run a simple E.R.O.S functionality test
  monitor     - Launch the interactive system monitor
  dashboard   - Launch the dashboard with mock data
  readiness   - Check E.R.O.S system readiness and configuration
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['gui', 'ultimate', 'simple', 'monitor', 'dashboard', 'readiness'],
        help='Operation mode to launch'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Duration in seconds for autonomous modes (default: 300)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        from src.dashboard.core import create_interactive_dashboard
        from src.core.logging_config import InstrumentationCollector
        
        print("🎯 Launching E.R.O.S Control Center...")
        collector = InstrumentationCollector()
        dashboard = create_interactive_dashboard(collector, update_interval=1.0)
        dashboard.run()
        
    elif args.mode == 'ultimate':
        from src.agents.ultimate_agent import UltimateAutonomousAgent
        
        agent = UltimateAutonomousAgent(duration_minutes=args.duration / 60.0)
        agent.run()
        
    elif args.mode == 'simple':
        from src.agents.simple_agent import SimpleAgent
        
        agent = SimpleAgent()
        agent.run(cycles=10, delay=1.0)
        
    elif args.mode == 'monitor':
        from demos.interactive_agent_monitor import main as monitor_main
        monitor_main()
        
    elif args.mode == 'dashboard':
        from tests.test_interactive_dashboard import main as dashboard_main
        dashboard_main()
        
    elif args.mode == 'readiness':
        from utils.system_readiness_check import main as readiness_main
        readiness_main()

if __name__ == "__main__":
    main()