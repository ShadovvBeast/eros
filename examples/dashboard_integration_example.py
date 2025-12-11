#!/usr/bin/env python3
"""
Dashboard Integration Example

This example shows how to integrate the interactive dashboard with your existing agent code.
"""

import sys
import os
sys.path.append('../src')

import threading
import time
import numpy as np
from datetime import datetime

from src.interactive_dashboard import create_interactive_dashboard
from src.logging_config import InstrumentationCollector
from src.agent import AutonomousAgent
from src.config import AgentConfig

class DashboardIntegrationExample:
    """
    Example showing how to integrate the interactive dashboard with an existing agent.
    """
    
    def __init__(self):
        self.collector = InstrumentationCollector()
        self.dashboard = None
        self.agent = None
        self.is_running = False
    
    def setup_agent_with_monitoring(self):
        """Setup agent with comprehensive monitoring"""
        print("🤖 Setting up agent with monitoring...")
        
        # Create agent configuration
        config = AgentConfig()
        
        # Initialize agent with instrumentation
        self.agent = AutonomousAgent(config, instrumentation=self.collector)
        
        print("✅ Agent setup complete with monitoring enabled")
    
    def setup_dashboard(self):
        """Setup the interactive dashboard"""
        print("📊 Setting up interactive dashboard...")
        
        # Create dashboard with 1-second updates for demo
        self.dashboard = create_interactive_dashboard(
            self.collector, 
            update_interval=1.0
        )
        
        print("✅ Dashboard setup complete")
    
    def simulate_agent_activity(self, duration_seconds: int = 60):
        """Simulate agent activity for demonstration"""
        print(f"🚀 Starting agent simulation for {duration_seconds} seconds...")
        
        def agent_simulation():
            self.is_running = True
            start_time = time.time()
            cycle = 0
            
            while (time.time() - start_time) < duration_seconds and self.is_running:
                cycle += 1
                
                # Simulate agent cycle
                self._simulate_agent_cycle(cycle)
                
                # Wait between cycles
                time.sleep(0.2)  # 5 cycles per second
            
            print(f"🏁 Agent simulation completed {cycle} cycles")
            self.is_running = False
        
        # Start simulation in background thread
        simulation_thread = threading.Thread(target=agent_simulation, daemon=True)
        simulation_thread.start()
        
        return simulation_thread
    
    def _simulate_agent_cycle(self, cycle: int):
        """Simulate a single agent cycle with realistic data"""
        
        # Simulate cycle timing with some variation
        base_time = 0.2
        variation = 0.05 * np.sin(cycle * 0.1) + 0.02 * np.random.randn()
        cycle_time = max(0.1, base_time + variation)
        
        # Record cycle timing
        self.collector.metrics.setdefault('cycle_timings', []).append({
            'cycle': cycle,
            'total_duration': cycle_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Simulate tool usage
        tools = [
            'creative_problem_solver', 'analysis_data_processor', 
            'communication_enhancer', 'system_optimizer', 'learning_synthesizer'
        ]
        
        if cycle % 3 == 0:  # Use tool every 3rd cycle
            tool = np.random.choice(tools)
            success = np.random.random() > 0.1  # 90% success rate
            
            self.collector.metrics.setdefault('tool_usage', []).append({
                'tool_name': tool,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
        
        # Simulate memory events
        if cycle % 7 == 0:  # Memory event every 7th cycle
            salience = 0.3 + 0.4 * np.random.random()  # Salience between 0.3-0.7
            
            self.collector.metrics.setdefault('memory_events', []).append({
                'event_type': 'store',
                'salience': salience,
                'timestamp': datetime.now().isoformat()
            })
        
        # Simulate attractor patterns
        attractor_count = max(1, int(3 + 2 * np.sin(cycle * 0.02)))
        strength = 0.4 + 0.3 * np.sin(cycle * 0.05) + 0.1 * np.random.randn()
        
        self.collector.metrics.setdefault('attractor_patterns', []).append({
            'cycle': cycle,
            'attractor_count': attractor_count,
            'strongest_attractor_strength': max(0, min(1, strength)),
            'pattern_recognized': np.random.random() > 0.6
        })
        
        # Simulate preference updates
        if cycle % 15 == 0:  # Update preferences every 15 cycles
            categories = ['exploration', 'exploitation', 'creativity', 'analysis', 'communication']
            
            for category in categories:
                if category not in self.collector.preference_history:
                    self.collector.preference_history[category] = []
                
                # Simulate preference drift
                base_weight = 0.5
                drift = 0.2 * np.sin(cycle * 0.01) + 0.05 * np.random.randn()
                weight = max(0.1, min(0.9, base_weight + drift))
                
                self.collector.preference_history[category].append((
                    datetime.now().isoformat(), weight
                ))
                
                # Keep only recent history
                if len(self.collector.preference_history[category]) > 50:
                    self.collector.preference_history[category] = \
                        self.collector.preference_history[category][-50:]
        
        # Simulate Pathos states
        if not hasattr(self.collector, 'pathos_states'):
            self.collector.pathos_states = []
        
        # Create realistic Pathos state
        state_dim = 64
        base_state = np.sin(cycle * 0.02) * np.ones(state_dim)
        noise = 0.1 * np.random.randn(state_dim)
        state_vector = base_state + noise
        
        internal_reward = np.sin(cycle * 0.03) + 0.2 * np.random.randn()
        
        self.collector.pathos_states.append({
            'state_vector': state_vector,
            'internal_reward': internal_reward,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent states
        if len(self.collector.pathos_states) > 100:
            self.collector.pathos_states = self.collector.pathos_states[-100:]
        
        # Occasionally simulate errors
        if np.random.random() < 0.02:  # 2% error rate
            error_types = ['timeout', 'validation', 'network', 'resource']
            self.collector.metrics.setdefault('error_events', []).append({
                'error_type': np.random.choice(error_types),
                'timestamp': datetime.now().isoformat()
            })
    
    def run_integrated_example(self, simulation_duration: int = 120):
        """Run the complete integrated example"""
        try:
            print("🎯 DASHBOARD INTEGRATION EXAMPLE")
            print("=" * 50)
            
            # Setup components
            self.setup_agent_with_monitoring()
            self.setup_dashboard()
            
            print(f"\n🚀 Starting integrated system:")
            print(f"   Simulation Duration: {simulation_duration} seconds")
            print(f"   Dashboard Updates: Every 1 second")
            print(f"   Agent Cycles: ~5 per second")
            
            # Start agent simulation
            simulation_thread = self.simulate_agent_activity(simulation_duration)
            
            print("\n💡 Dashboard Instructions:")
            print("   • Click 'Start Monitoring' to see live updates")
            print("   • Switch between tabs to see different visualizations")
            print("   • Use 'Export Data' to save current state")
            print("   • Close the dashboard window to stop the example")
            
            # Run dashboard (this blocks until window is closed)
            self.dashboard.run()
            
        except KeyboardInterrupt:
            print("\n🛑 Example interrupted by user")
        except Exception as e:
            print(f"\n❌ Example error: {e}")
        finally:
            # Cleanup
            self.is_running = False
            print("🧹 Example cleanup complete")
    
    def stop(self):
        """Stop the example"""
        self.is_running = False


def main():
    """Main entry point"""
    print("🎯 DASHBOARD INTEGRATION EXAMPLE")
    print("=" * 50)
    print("This example demonstrates how to integrate the interactive dashboard")
    print("with your autonomous agent system for real-time monitoring.")
    print()
    
    # Create and run example
    example = DashboardIntegrationExample()
    
    try:
        # Ask user for simulation duration
        duration_input = input("Enter simulation duration in seconds (default 120): ").strip()
        duration = int(duration_input) if duration_input else 120
        
        example.run_integrated_example(duration)
        
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except ValueError:
        print("❌ Invalid duration, using default (120 seconds)")
        example.run_integrated_example(120)
    except Exception as e:
        print(f"❌ Example error: {e}")


if __name__ == "__main__":
    main()