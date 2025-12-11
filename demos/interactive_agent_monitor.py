#!/usr/bin/env python3
"""
Interactive Agent Monitor

Launch the autonomous agent with a unified interactive dashboard for real-time monitoring.
This provides a professional, tabbed interface for comprehensive agent analysis.
"""

import sys
import os
import threading
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.interactive_dashboard import create_interactive_dashboard
from src.logging_config import InstrumentationCollector
from src.agent import AutonomousAgent
from src.config import AgentConfig
from src.models import MemoryTrace

class InteractiveAgentMonitor:
    """
    Integrated system that runs the autonomous agent with real-time dashboard monitoring.
    """
    
    def __init__(self):
        self.agent = None
        self.collector = None
        self.dashboard = None
        self.agent_thread = None
        self.is_agent_running = False
        
    def setup_agent(self, config_name: str = "default"):
        """Setup the autonomous agent with monitoring"""
        print("🤖 Setting up Autonomous Agent with Interactive Dashboard...")
        
        # Create configuration
        config = AgentConfig()
        
        # Initialize instrumentation collector
        self.collector = InstrumentationCollector()
        
        # Initialize agent
        self.agent = AutonomousAgent(config, instrumentation=self.collector)
        
        print("✅ Agent setup complete!")
        
    def setup_dashboard(self, update_interval: float = 2.0):
        """Setup the interactive dashboard"""
        print("📊 Setting up Interactive Dashboard...")
        
        if not self.collector:
            raise ValueError("Agent must be setup before dashboard")
        
        # Create dashboard
        self.dashboard = create_interactive_dashboard(
            self.collector, 
            update_interval=update_interval
        )
        
        print("✅ Dashboard setup complete!")
        
    def start_agent_background(self, duration_minutes: int = 10):
        """Start the agent in background thread"""
        if self.is_agent_running:
            print("⚠️ Agent is already running")
            return
        
        print(f"🚀 Starting agent for {duration_minutes} minutes...")
        
        def agent_runner():
            self.is_agent_running = True
            try:
                # Run agent cycles
                start_time = time.time()
                cycle = 0
                
                while (time.time() - start_time) < (duration_minutes * 60) and self.is_agent_running:
                    cycle += 1
                    
                    # Agent cycle
                    try:
                        # Simulate agent thinking and action
                        thought = f"Cycle {cycle}: Analyzing current state and planning next action..."
                        
                        # Update collector with mock data for demonstration
                        self._update_mock_data(cycle)
                        
                        # Small delay between cycles
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Agent cycle {cycle} error: {e}")
                        
                print(f"🏁 Agent completed {cycle} cycles")
                
            except Exception as e:
                print(f"Agent runner error: {e}")
            finally:
                self.is_agent_running = False
        
        # Start agent in background thread
        self.agent_thread = threading.Thread(target=agent_runner, daemon=True)
        self.agent_thread.start()
        
    def _update_mock_data(self, cycle: int):
        """Update collector with mock data for demonstration"""
        import numpy as np
        from datetime import datetime
        
        # Mock cycle timing
        self.collector.metrics.setdefault('cycle_timings', []).append({
            'cycle': cycle,
            'total_duration': 0.5 + 0.2 * np.sin(cycle * 0.1),
            'timestamp': datetime.now().isoformat()
        })
        
        # Mock tool usage
        tools = ['creative_problem_solver', 'analysis_data_processor', 'communication_enhancer']
        tool = np.random.choice(tools)
        success = np.random.random() > 0.2  # 80% success rate
        
        self.collector.metrics.setdefault('tool_usage', []).append({
            'tool_name': tool,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mock memory events
        if cycle % 5 == 0:  # Memory event every 5 cycles
            self.collector.metrics.setdefault('memory_events', []).append({
                'event_type': 'store',
                'salience': np.random.random(),
                'timestamp': datetime.now().isoformat()
            })
        
        # Mock attractor patterns
        self.collector.metrics.setdefault('attractor_patterns', []).append({
            'cycle': cycle,
            'attractor_count': max(1, int(3 + 2 * np.sin(cycle * 0.05))),
            'strongest_attractor_strength': 0.5 + 0.3 * np.sin(cycle * 0.03),
            'pattern_recognized': np.random.random() > 0.7
        })
        
        # Mock preference history
        if cycle % 10 == 0:  # Update preferences every 10 cycles
            categories = ['exploration', 'exploitation', 'creativity', 'analysis']
            for category in categories:
                if category not in self.collector.preference_history:
                    self.collector.preference_history[category] = []
                
                # Add some drift over time
                base_weight = 0.5
                drift = 0.2 * np.sin(cycle * 0.02) + 0.1 * np.random.random()
                weight = max(0.1, min(0.9, base_weight + drift))
                
                self.collector.preference_history[category].append((
                    datetime.now().isoformat(), weight
                ))
        
        # Mock Pathos states
        if not hasattr(self.collector, 'pathos_states'):
            self.collector.pathos_states = []
        
        state_vector = np.random.randn(64) * 0.1 + np.sin(cycle * 0.1)
        internal_reward = np.sin(cycle * 0.05) + 0.1 * np.random.randn()
        
        self.collector.pathos_states.append({
            'state_vector': state_vector,
            'internal_reward': internal_reward,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent states
        if len(self.collector.pathos_states) > 100:
            self.collector.pathos_states = self.collector.pathos_states[-100:]
    
    def stop_agent(self):
        """Stop the agent"""
        if self.is_agent_running:
            print("🛑 Stopping agent...")
            self.is_agent_running = False
            if self.agent_thread:
                self.agent_thread.join(timeout=2)
            print("✅ Agent stopped")
    
    def run_dashboard(self):
        """Run the interactive dashboard"""
        if not self.dashboard:
            raise ValueError("Dashboard must be setup before running")
        
        print("🎯 Launching Interactive Dashboard...")
        print("💡 Use the dashboard controls to start/stop monitoring")
        print("📊 All visualizations will update in real-time")
        
        # Run dashboard (this blocks until window is closed)
        self.dashboard.run()
    
    def run_integrated_system(self, agent_duration: int = 10, update_interval: float = 2.0):
        """Run the complete integrated system"""
        try:
            # Setup components
            self.setup_agent()
            self.setup_dashboard(update_interval)
            
            # Start agent in background
            self.start_agent_background(agent_duration)
            
            # Run dashboard (blocks until closed)
            self.run_dashboard()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            # Cleanup
            self.stop_agent()
            print("🧹 Cleanup complete")


def main():
    """Main entry point"""
    print("🎯 INTERACTIVE AUTONOMOUS AGENT MONITOR")
    print("=" * 60)
    print("🚀 Unified Dashboard for Real-Time Agent Monitoring")
    print("📊 Professional Tabbed Interface")
    print("🔄 Live Updates and Analysis")
    print("=" * 60)
    
    # Create monitor
    monitor = InteractiveAgentMonitor()
    
    # Configuration menu
    print("\n🎮 Configuration Options:")
    print("1. Quick Demo (5 minutes)")
    print("2. Standard Session (15 minutes)")
    print("3. Extended Session (30 minutes)")
    print("4. Custom Duration")
    print("5. Dashboard Only (no agent)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            duration = 5
            interval = 1.0
        elif choice == "2":
            duration = 15
            interval = 2.0
        elif choice == "3":
            duration = 30
            interval = 3.0
        elif choice == "4":
            duration = int(input("Enter duration in minutes: "))
            interval = float(input("Enter update interval in seconds (1-10): "))
        elif choice == "5":
            # Dashboard only mode
            monitor.setup_agent()
            monitor.setup_dashboard(2.0)
            monitor.run_dashboard()
            return
        else:
            print("Invalid choice, using defaults")
            duration = 10
            interval = 2.0
        
        print(f"\n🚀 Starting integrated system:")
        print(f"   Agent Duration: {duration} minutes")
        print(f"   Update Interval: {interval} seconds")
        print(f"   Dashboard: Real-time monitoring")
        
        # Run integrated system
        monitor.run_integrated_system(duration, interval)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()