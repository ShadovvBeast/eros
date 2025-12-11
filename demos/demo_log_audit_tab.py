#!/usr/bin/env python3
"""
Demo: Log & Audit Tab

Demonstrates the new log and audit functionality by generating
various types of system events and showing how they appear in the log.
"""

import sys
import time
import threading
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.interactive_dashboard import InteractiveDashboard
from src.logging_config import InstrumentationCollector
from src.session_manager import get_session_manager, SessionConfig
from src.thread_manager import get_thread_manager


def demo_log_audit_features():
    """Demonstrate log and audit tab features"""
    print("🎬 LOG & AUDIT TAB DEMO")
    print("=" * 60)
    
    # Create dashboard
    collector = InstrumentationCollector()
    dashboard = InteractiveDashboard(collector, update_interval=1.0)
    
    print("📋 Dashboard created with Log & Audit tab")
    print("   The dashboard now includes a '📋 Log & Audit' tab with:")
    print("   • Real-time system event logging")
    print("   • Event filtering and search")
    print("   • Statistics and export capabilities")
    print()
    
    # Demonstrate different log levels
    print("🎯 Demonstrating Log Levels...")
    dashboard._add_log_entry("DEMO", "DEBUG", "Debug message example", "Detailed debugging info")
    dashboard._add_log_entry("DEMO", "INFO", "Information message", "General system information")
    dashboard._add_log_entry("DEMO", "WARNING", "Warning message", "Something needs attention")
    dashboard._add_log_entry("DEMO", "ERROR", "Error message", "Something went wrong")
    dashboard._add_log_entry("DEMO", "CRITICAL", "Critical message", "System critical issue")
    
    print("   ✅ Added examples of all log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    print()
    
    # Demonstrate different components
    print("🔧 Demonstrating Component Logging...")
    components = ["SESSION", "AGENT", "TOOLS", "MEMORY", "PATHOS", "ETHOS", "HARDWARE", "THREADS"]
    
    for i, component in enumerate(components):
        dashboard._add_log_entry(component, "INFO", f"{component} component active", f"Component {i+1} status update")
    
    print(f"   ✅ Added log entries for {len(components)} different system components")
    print()
    
    # Demonstrate session integration
    print("🎮 Demonstrating Session Integration...")
    sm = get_session_manager()
    config = SessionConfig()
    config.duration_minutes = 0.15  # 9 seconds
    config.max_cycles = 3
    # No cycle delay - maximum efficiency mode
    
    sm.configure_session(config)
    
    initial_count = dashboard.event_counts['total']
    print(f"   Initial event count: {initial_count}")
    
    # Start session
    print("   Starting agent session...")
    success = sm.start_session()
    
    if success:
        print("   ✅ Session started - watch for session events in log")
        
        # Let session run
        time.sleep(5)
        
        # Check event count
        mid_count = dashboard.event_counts['total']
        print(f"   Mid-session event count: {mid_count} (+{mid_count - initial_count} new events)")
        
        # Wait for completion
        time.sleep(6)
        sm.stop_session()
        
        final_count = dashboard.event_counts['total']
        print(f"   Final event count: {final_count} (+{final_count - initial_count} total new events)")
    else:
        print("   ❌ Session failed to start")
    
    print()
    
    # Demonstrate thread activity logging
    print("🧵 Demonstrating Thread Activity Logging...")
    tm = get_thread_manager()
    
    def demo_worker(name, duration):
        dashboard._add_log_entry("WORKER", "INFO", f"Worker {name} started", f"Duration: {duration}s")
        time.sleep(duration)
        dashboard._add_log_entry("WORKER", "INFO", f"Worker {name} completed", "Task finished successfully")
    
    # Create worker threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=demo_worker, args=(f"Demo-{i}", 2), name=f"DemoWorker-{i}")
        tm.register_thread(thread, "demo", "LogDemo")
        threads.append(thread)
        thread.start()
    
    print("   ✅ Started 3 demo worker threads")
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    print("   ✅ All worker threads completed")
    print()
    
    # Show final statistics
    print("📊 Final Log Statistics:")
    print(f"   Total Events: {dashboard.event_counts['total']}")
    print(f"   Session Events: {dashboard.event_counts['session']}")
    print(f"   Tool Events: {dashboard.event_counts['tool']}")
    print(f"   Error Events: {dashboard.event_counts['error']}")
    print(f"   Thread Events: {dashboard.event_counts['thread']}")
    print(f"   Buffer Size: {len(dashboard.log_buffer)} entries")
    
    if dashboard.last_event_time:
        print(f"   Last Event: {dashboard.last_event_time.strftime('%H:%M:%S')}")
    
    print()
    print("🎉 LOG & AUDIT TAB DEMO COMPLETE!")
    print()
    print("📋 To see the log in action:")
    print("   1. Run: python autonomous_agent_gui.py")
    print("   2. Click '📊 Start Monitoring'")
    print("   3. Navigate to the '📋 Log & Audit' tab")
    print("   4. Start a session to see real-time logging")
    print()
    print("🔍 Log Tab Features:")
    print("   • Real-time event stream with color coding")
    print("   • Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    print("   • Filter by component (Session, Agent, Tools, etc.)")
    print("   • Search functionality")
    print("   • Event statistics and rate monitoring")
    print("   • Export to log files or JSON statistics")
    print("   • Auto-scroll and buffer size controls")
    print("   • Copy to clipboard functionality")


if __name__ == "__main__":
    demo_log_audit_features()