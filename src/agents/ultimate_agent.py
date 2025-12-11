"""
Ultimate Autonomous Agent

Advanced autonomous agent with time-based operation and full capabilities.
"""

import time
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from ..core.agent import AutonomousAgent
    from ..core.config import AgentConfig
    from ..core.logging_config import InstrumentationCollector
    from ..visualization import (
        PathosStateVisualizer, 
        PreferenceDriftAnalyzer,
        AttractorPatternDetector,
        MemoryNetworkVisualizer,
        SystemHealthDashboard
    )
except ImportError:
    from core.agent import AutonomousAgent
    from core.config import AgentConfig
    from logging_config import InstrumentationCollector
    from visualization import (
        PathosStateVisualizer, 
        PreferenceDriftAnalyzer,
        AttractorPatternDetector,
        MemoryNetworkVisualizer,
        SystemHealthDashboard
    )


class UltimateAutonomousAgent:
    """Ultimate autonomous agent with time-based operation and full capabilities"""
    
    def __init__(self, duration_minutes: float = 5.0):
        """Initialize ultimate agent."""
        self.duration_minutes = duration_minutes
        self.config = AgentConfig()
        self.collector = InstrumentationCollector()
        self.agent = None
        self.is_running = False
        self.start_time = None
        self.end_time = None
        
        # Visualization components
        self.visualizers = {}
        self.output_directory = None
        
        # Performance tracking
        self.cycle_count = 0
        self.total_runtime = 0
        self.performance_metrics = []
        
        # Import numpy for calculations
        import numpy as np
        self.np = np
    
    def initialize(self):
        """Initialize the ultimate agent."""
        print("🎯 ULTIMATE AUTONOMOUS AGENT")
        print("=" * 60)
        print(f"⏱️  Duration: {self.duration_minutes} minutes")
        print(f"🔧 Full tool access enabled")
        print(f"📊 Comprehensive monitoring active")
        print("=" * 60)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_directory = f"ultimate_agent_output_{timestamp}"
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Initialize agent
        self.agent = AutonomousAgent(
            config=self.config,
            instrumentation=self.collector
        )
        
        # Initialize layers
        from ..logos.logos_layer import LogosLayer
        from ..pathos.pathos_layer import PathosLayer
        from ..memory.memory_system import ConcreteMemorySystem
        from ..ethos.ethos_framework import ConcreteEthosFramework
        from ..tools.tool_layer import ToolLayer
        
        logos = LogosLayer(self.config.logos)
        pathos = PathosLayer(self.config.pathos)
        memory = ConcreteMemorySystem(self.config.memory)
        ethos = ConcreteEthosFramework(self.config.ethos)
        tools = ToolLayer(self.config.tools)
        
        self.agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        # Initialize visualizers
        self._initialize_visualizers()
        
        print("✅ Ultimate agent initialized successfully")
    
    def _initialize_visualizers(self):
        """Initialize visualization components."""
        try:
            self.visualizers = {
                'pathos': PathosStateVisualizer(),
                'preference': PreferenceDriftAnalyzer(),
                'attractor': AttractorPatternDetector(),
                'memory': MemoryNetworkVisualizer(),
                'health': SystemHealthDashboard()
            }
            print("📊 Visualization components initialized")
        except Exception as e:
            print(f"⚠️ Visualization initialization warning: {e}")
    
    def run(self):
        """Run the ultimate agent for the specified duration."""
        if not self.agent:
            self.initialize()
        
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        self.is_running = True
        
        print(f"\n🚀 Starting ultimate agent operation...")
        print(f"📅 Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitor_thread.start()
            
            # Main operation loop
            while self.is_running and datetime.now() < self.end_time:
                cycle_start = time.time()
                
                # Run agent cycle
                try:
                    result = self.agent.run_cycle()
                    self.cycle_count += 1
                    
                    cycle_time = time.time() - cycle_start
                    self.performance_metrics.append({
                        'cycle': self.cycle_count,
                        'duration': cycle_time,
                        'timestamp': datetime.now().isoformat(),
                        'success': True
                    })
                    
                    # Update visualizations
                    self._update_visualizations()
                    
                    # Progress update
                    if self.cycle_count % 10 == 0:
                        elapsed = datetime.now() - self.start_time
                        remaining = self.end_time - datetime.now()
                        print(f"🔄 Cycle {self.cycle_count} | "
                              f"Elapsed: {elapsed.total_seconds():.1f}s | "
                              f"Remaining: {remaining.total_seconds():.1f}s")
                
                except Exception as e:
                    print(f"❌ Cycle {self.cycle_count + 1} failed: {e}")
                    self.performance_metrics.append({
                        'cycle': self.cycle_count + 1,
                        'duration': time.time() - cycle_start,
                        'timestamp': datetime.now().isoformat(),
                        'success': False,
                        'error': str(e)
                    })
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            self._finalize_operation()
            
        except KeyboardInterrupt:
            print("\n⏹️ Ultimate agent operation interrupted by user")
            self._finalize_operation()
        except Exception as e:
            print(f"\n❌ Ultimate agent operation failed: {e}")
            self._finalize_operation()
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                # Update health dashboard
                if 'health' in self.visualizers:
                    self.visualizers['health'].update_health_metrics(self.collector)
                
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(1)
    
    def _update_visualizations(self):
        """Update visualization components."""
        try:
            # Update pathos visualizer
            if 'pathos' in self.visualizers and hasattr(self.agent, 'pathos_layer'):
                if hasattr(self.agent.pathos_layer, 'current_state'):
                    state = self.agent.pathos_layer.current_state.state_vector
                    reward = getattr(self.agent.pathos_layer.current_state, 'internal_reward', 0.0)
                    self.visualizers['pathos'].update_state(state, reward)
        except Exception as e:
            # Silently handle visualization errors to not interrupt main operation
            pass
    
    def _finalize_operation(self):
        """Finalize the operation and generate reports."""
        self.is_running = False
        actual_end_time = datetime.now()
        self.total_runtime = (actual_end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("🏁 ULTIMATE AGENT OPERATION COMPLETE")
        print("=" * 60)
        
        # Performance summary
        successful_cycles = sum(1 for m in self.performance_metrics if m.get('success', False))
        failed_cycles = len(self.performance_metrics) - successful_cycles
        avg_cycle_time = self.np.mean([m['duration'] for m in self.performance_metrics]) if self.performance_metrics else 0
        
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Total Runtime: {self.total_runtime:.2f} seconds")
        print(f"   Total Cycles: {len(self.performance_metrics)}")
        print(f"   Successful Cycles: {successful_cycles}")
        print(f"   Failed Cycles: {failed_cycles}")
        print(f"   Average Cycle Time: {avg_cycle_time:.3f} seconds")
        print(f"   Cycles per Second: {len(self.performance_metrics) / self.total_runtime:.2f}")
        
        # Generate reports
        self._generate_reports()
        
        print(f"\n📁 Output Directory: {self.output_directory}")
        print("✅ All reports and visualizations saved")
    
    def _generate_reports(self):
        """Generate comprehensive reports and visualizations."""
        try:
            # Performance report
            performance_file = os.path.join(self.output_directory, "performance_report.json")
            with open(performance_file, 'w') as f:
                json.dump({
                    'summary': {
                        'total_runtime': self.total_runtime,
                        'total_cycles': len(self.performance_metrics),
                        'successful_cycles': sum(1 for m in self.performance_metrics if m.get('success', False)),
                        'average_cycle_time': self.np.mean([m['duration'] for m in self.performance_metrics]) if self.performance_metrics else 0
                    },
                    'detailed_metrics': self.performance_metrics
                }, f, indent=2)
            
            # Save visualizations
            for name, visualizer in self.visualizers.items():
                try:
                    if hasattr(visualizer, 'save_visualization'):
                        viz_file = os.path.join(self.output_directory, f"{name}_visualization.png")
                        visualizer.save_visualization(viz_file)
                    elif hasattr(visualizer, 'visualize_preference_drift'):
                        viz_file = os.path.join(self.output_directory, f"{name}_analysis.png")
                        visualizer.visualize_preference_drift(self.collector, viz_file)
                except Exception as e:
                    print(f"⚠️ Could not save {name} visualization: {e}")
            
            # Instrumentation data
            if hasattr(self.collector, 'export_data'):
                data_file = os.path.join(self.output_directory, "instrumentation_data.json")
                self.collector.export_data(data_file)
            
        except Exception as e:
            print(f"⚠️ Report generation error: {e}")
    
    def stop(self):
        """Stop the ultimate agent."""
        self.is_running = False
        print("🛑 Ultimate agent stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'total_runtime': self.total_runtime,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'output_directory': self.output_directory,
            'performance_metrics_count': len(self.performance_metrics)
        }


def main():
    """Main entry point for ultimate agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Autonomous Agent")
    parser.add_argument('--duration', type=float, default=5.0, 
                       help='Duration in minutes (default: 5.0)')
    
    args = parser.parse_args()
    
    # Create and run ultimate agent
    agent = UltimateAutonomousAgent(duration_minutes=args.duration)
    agent.run()


if __name__ == "__main__":
    main()