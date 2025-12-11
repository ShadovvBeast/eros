#!/usr/bin/env python3
"""
Autonomous Agent GUI Control Center

The primary interface for the autonomous agent system. Provides comprehensive
session management, real-time monitoring, and hardware tracking through a
professional GUI interface.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.interactive_dashboard import create_interactive_dashboard


def main():
    """Main entry point for the Autonomous Agent GUI"""
    print("🎯 AUTONOMOUS AGENT CONTROL CENTER")
    print("=" * 60)
    print("🚀 Professional GUI Interface for Agent Management")
    print("📊 Real-time Monitoring & Hardware Tracking")
    print("🔧 Comprehensive Session Control")
    print("💾 Advanced Data Export & Analysis")
    print("=" * 60)
    
    try:
        # Create the interactive dashboard
        print("🚀 Initializing Control Center...")
        
        # Create a collector for the dashboard
        from src.logging_config import InstrumentationCollector
        collector = InstrumentationCollector()
        
        dashboard = create_interactive_dashboard(collector, update_interval=1.0)
        
        print("✅ Control Center ready!")
        print("\n💡 Getting Started:")
        print("   1. Configure your agent using the '⚙️ Configure Agent' button")
        print("   2. Set session duration and parameters")
        print("   3. Click '▶️ Start Session' to begin agent operation")
        print("   4. Use '📊 Start Monitoring' for real-time visualization")
        print("   5. Monitor hardware usage in the Hardware Monitor tab")
        print("   6. Export data and analysis using the '💾 Export Data' button")
        
        print("\n🎮 Control Center launching...")
        
        # Run the dashboard
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\n👋 Control Center shutdown requested by user")
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check the error details above and try again")
        sys.exit(1)
    
    print("🧹 Control Center shutdown complete")


if __name__ == "__main__":
    main()