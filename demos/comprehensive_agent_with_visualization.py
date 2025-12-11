#!/usr/bin/env python3
"""
Comprehensive Autonomous Agent Test with Full Visualization and Monitoring

This script demonstrates the autonomous agent with complete visualization,
monitoring, and analysis capabilities including:
- Real-time Pathos state visualization
- Preference drift analysis
- Attractor pattern detection
- Memory network visualization
- System health dashboard
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agent import AutonomousAgent
from src.logos.logos_layer import LogosLayer
from src.pathos.pathos_layer import PathosLayer
from src.memory.memory_system import ConcreteMemorySystem
from src.ethos.ethos_framework import ConcreteEthosFramework
from src.tools.tool_layer import ToolLayer
from src.config import AgentConfig, PathosConfig, LogosConfig, MemoryConfig, EthosConfig
from src.logging_config import InstrumentationCollector, AgentLogger
from src.visualization import (
    PathosStateVisualizer, 
    PreferenceDriftAnalyzer,
    AttractorPatternDetector,
    MemoryNetworkVisualizer,
    SystemHealthDashboard,
    create_comprehensive_analysis
)


class ComprehensiveAgentDemo:
    """Comprehensive agent demonstration with full monitoring and visualization"""
    
    def __init__(self):
        self.agent = None
        self.instrumentation = None
        self.pathos_visualizer = None
        self.output_dir = Path("agent_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load environment variables from .env file"""
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    def create_enhanced_config(self):
        """Create enhanced configuration for comprehensive testing"""
        
        # Ethos configuration - curious and exploratory
        ethos_config = EthosConfig(
            identity_description="""
            I am a comprehensive autonomous agent designed for exploration and learning.
            I am equipped with advanced monitoring and visualization capabilities.
            My goal is to demonstrate autonomous behavior while providing rich insights
            into my internal dynamics, preferences, and decision-making processes.
            I am curious, analytical, and focused on continuous improvement.
            """,
            core_values=[
                "curiosity",
                "learning", 
                "exploration",
                "analysis",
                "growth",
                "transparency",
                "safety"
            ],
            safety_constraints=[
                "no_harmful_actions",
                "respect_privacy",
                "stay_within_bounds",
                "maintain_transparency"
            ],
            allowed_tool_categories=[
                "analysis",
                "communication",
                "web_search",
                "file_operations"
            ],
            personality_traits={
                'openness': 0.9,
                'conscientiousness': 0.8,
                'extraversion': 0.6,
                'agreeableness': 0.8,
                'neuroticism': 0.2,
                'curiosity': 0.95,
                'creativity': 0.8,
                'analytical': 0.9,
                'caution': 0.5
            }
        )
        
        # Pathos configuration - optimized for interesting dynamics
        pathos_config = PathosConfig(
            state_dimension=128,
            decay_factor=0.92,  # Slightly more dynamic
            echo_strength=0.15,  # Stronger memory influence
            homeostatic_weights={
                'stability': 0.8,
                'complexity': 0.7,
                'coherence': 0.9,
                'exploration': 0.6
            },
            salience_threshold=0.5,  # Lower threshold for more memories
            memory_retrieval_k=5,
            # Enhanced salience coefficients for richer dynamics
            c1=1.2,  # State change magnitude
            c2=1.0,  # Reward magnitude
            c3=1.1,  # Novelty (affect)
            c4=0.9,  # Novelty (semantic)
            c5=0.8   # Interest signal
        )
        
        # Logos configuration
        logos_config = LogosConfig(
            semantic_dimension=256,
            max_intentions_per_cycle=3,
            planning_horizon=5,
            interest_threshold=0.4,  # Lower threshold for more engagement
            gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
            gemini_model='gemini-2.5-flash'
        )
        
        # Memory configuration
        memory_config = MemoryConfig(
            max_memory_traces=2000,
            similarity_threshold=0.6,  # Lower threshold for more connections
            novelty_k_neighbors=8,
            storage_backend='memory'
        )
        
        config = AgentConfig(
            ethos=ethos_config,
            pathos=pathos_config,
            logos=logos_config,
            memory=memory_config
        )
        
        # Enhanced monitoring settings
        config.cycle_interval_seconds = 0.8  # Faster cycles for more data
        config.enable_monitoring = True
        config.log_level = "INFO"
        
        return config
    
    def create_enhanced_tools(self):
        """Create enhanced tools for comprehensive testing"""
        
        class AnalysisToolEnhanced:
            @property
            def name(self) -> str:
                return "data_analyzer"
            
            @property
            def description(self) -> str:
                return "Analyze data patterns, trends, and generate insights"
            
            @property
            def category(self) -> str:
                return "analysis"
            
            def execute(self, arguments: Dict[str, Any]):
                from src.models import ToolResult
                
                try:
                    data_type = arguments.get('type', 'general')
                    complexity = arguments.get('complexity', 'medium')
                    
                    # Simulate different analysis outcomes
                    if data_type == 'pattern':
                        result = {
                            'patterns_found': np.random.randint(2, 8),
                            'confidence': np.random.uniform(0.6, 0.95),
                            'insights': [
                                "Cyclical pattern detected",
                                "Trend shows positive growth",
                                "Anomaly at data point 15"
                            ]
                        }
                    elif data_type == 'statistical':
                        result = {
                            'mean': np.random.uniform(10, 100),
                            'std_dev': np.random.uniform(1, 10),
                            'correlation': np.random.uniform(-1, 1),
                            'significance': np.random.uniform(0.01, 0.1)
                        }
                    else:
                        result = {
                            'analysis_type': data_type,
                            'complexity_level': complexity,
                            'processing_time': np.random.uniform(0.1, 2.0),
                            'quality_score': np.random.uniform(0.7, 0.98)
                        }
                    
                    return ToolResult(
                        success=True,
                        result=result
                    )
                except Exception as e:
                    return ToolResult(
                        success=False,
                        error_message=str(e)
                    )
            
            def validate_arguments(self, arguments: Dict[str, Any]):
                return True, None
        
        class CreativeGeneratorTool:
            @property
            def name(self) -> str:
                return "creative_generator"
            
            @property
            def description(self) -> str:
                return "Generate creative ideas, concepts, and solutions"
            
            @property
            def category(self) -> str:
                return "communication"
            
            def execute(self, arguments: Dict[str, Any]):
                from src.models import ToolResult
                
                try:
                    theme = arguments.get('theme', 'innovation')
                    count = arguments.get('count', 3)
                    
                    ideas = [
                        f"Innovative approach to {theme} using AI",
                        f"Creative solution combining {theme} with sustainability",
                        f"Novel perspective on {theme} through interdisciplinary lens",
                        f"Breakthrough concept in {theme} with social impact",
                        f"Revolutionary {theme} methodology"
                    ]
                    
                    selected_ideas = np.random.choice(ideas, min(count, len(ideas)), replace=False)
                    
                    return ToolResult(
                        success=True,
                        result={
                            'theme': theme,
                            'ideas': selected_ideas.tolist(),
                            'creativity_score': np.random.uniform(0.7, 0.95),
                            'originality': np.random.uniform(0.6, 0.9)
                        }
                    )
                except Exception as e:
                    return ToolResult(
                        success=False,
                        error_message=str(e)
                    )
            
            def validate_arguments(self, arguments: Dict[str, Any]):
                return True, None
        
        class KnowledgeExplorerTool:
            @property
            def name(self) -> str:
                return "knowledge_explorer"
            
            @property
            def description(self) -> str:
                return "Explore knowledge domains and discover connections"
            
            @property
            def category(self) -> str:
                return "analysis"
            
            def execute(self, arguments: Dict[str, Any]):
                from src.models import ToolResult
                
                try:
                    domain = arguments.get('domain', 'general')
                    depth = arguments.get('depth', 'medium')
                    
                    domains = ['science', 'technology', 'philosophy', 'arts', 'mathematics', 'psychology']
                    connections = np.random.choice(domains, 3, replace=False)
                    
                    return ToolResult(
                        success=True,
                        result={
                            'explored_domain': domain,
                            'depth_level': depth,
                            'connections_found': connections.tolist(),
                            'knowledge_score': np.random.uniform(0.6, 0.9),
                            'insights': [
                                f"Deep connection between {domain} and {connections[0]}",
                                f"Emerging patterns in {domain} research",
                                f"Novel applications of {domain} principles"
                            ]
                        }
                    )
                except Exception as e:
                    return ToolResult(
                        success=False,
                        error_message=str(e)
                    )
            
            def validate_arguments(self, arguments: Dict[str, Any]):
                return True, None
        
        return [
            AnalysisToolEnhanced(),
            CreativeGeneratorTool(),
            KnowledgeExplorerTool()
        ]
    
    def setup_agent_with_monitoring(self):
        """Set up agent with comprehensive monitoring and visualization"""
        
        print("🚀 Setting up Comprehensive Autonomous Agent with Visualization")
        print("=" * 70)
        
        try:
            # Create configuration
            print("⚙️ Creating enhanced agent configuration...")
            config = self.create_enhanced_config()
            
            # Initialize instrumentation
            print("📊 Initializing monitoring and instrumentation...")
            self.instrumentation = InstrumentationCollector()
            
            # Create agent
            print("🤖 Initializing autonomous agent...")
            self.agent = AutonomousAgent(config)
            
            # Create layers with instrumentation
            ethos = ConcreteEthosFramework(config.ethos)
            logos = LogosLayer(config.logos)
            pathos = PathosLayer(config.pathos)
            memory = ConcreteMemorySystem(config.memory)
            tools = ToolLayer(config.tools, ethos)
            
            # Register enhanced tools
            print("🔧 Registering enhanced tools...")
            enhanced_tools = self.create_enhanced_tools()
            for tool in enhanced_tools:
                try:
                    tools.register_tool(tool)
                    print(f"   ✅ Registered: {tool.name}")
                except Exception as e:
                    print(f"   ⚠️ Failed to register {tool.name}: {e}")
            
            # Initialize agent with instrumentation
            self.agent.initialize_layers(logos, pathos, memory, ethos, tools)
            
            # Set up instrumentation callbacks
            self._setup_instrumentation_callbacks()
            
            # Initialize real-time visualizer
            print("📈 Setting up real-time visualization...")
            self.pathos_visualizer = PathosStateVisualizer(
                state_dimension=config.pathos.state_dimension,
                history_length=200
            )
            
            print("✅ Comprehensive agent setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_instrumentation_callbacks(self):
        """Set up instrumentation callbacks for monitoring"""
        
        # Register callback for tool results
        def tool_result_callback(result):
            self.instrumentation.record_tool_usage(
                cycle=getattr(self.agent, 'cycle_count', 0),
                tool_name=result.tool_name,
                success=result.success,
                execution_time=0.1,  # Simplified
                result_size=len(str(result.result)) if result.result else 0
            )
        
        if self.agent and self.agent.tools:
            self.agent.tools.register_result_callback(tool_result_callback)
    
    def run_comprehensive_session(self, cycles: int = 20, visualization_interval: int = 5):
        """Run comprehensive session with real-time monitoring and periodic analysis"""
        
        if not self.agent:
            print("❌ Agent not initialized")
            return False
        
        print(f"\n🎯 Running Comprehensive Session ({cycles} cycles)")
        print("=" * 50)
        print("📊 Real-time monitoring active")
        print("📈 Visualization updates every", visualization_interval, "cycles")
        print("💾 Analysis saved to:", self.output_dir)
        print("-" * 50)
        
        session_start = time.time()
        cycle_results = []
        
        try:
            for cycle in range(cycles):
                cycle_start = time.time()
                
                print(f"\n🔄 Cycle {cycle + 1}/{cycles}")
                
                # Run agent cycle
                try:
                    result = self.agent.run_cycle()
                    cycle_results.append(result)
                    
                    # Record instrumentation data
                    self._record_cycle_data(cycle, result, cycle_start)
                    
                    # Update real-time visualization
                    if hasattr(self.agent, 'pathos') and self.pathos_visualizer:
                        current_state = self.agent.pathos.current_state
                        internal_reward = result.get('internal_reward', 0)
                        self.pathos_visualizer.update_state(current_state, internal_reward)
                    
                    # Display cycle results
                    self._display_cycle_results(cycle + 1, result)
                    
                    # Periodic comprehensive analysis
                    if (cycle + 1) % visualization_interval == 0:
                        print(f"\n📊 Generating analysis at cycle {cycle + 1}...")
                        self._generate_periodic_analysis(cycle + 1)
                    
                    # Brief pause for visualization
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"   ❌ Cycle {cycle + 1} failed: {e}")
                    continue
            
            # Final comprehensive analysis
            print(f"\n🎉 Session Complete! Generating final analysis...")
            self._generate_final_analysis(cycles, session_start, cycle_results)
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Session interrupted by user at cycle {cycle + 1}")
            self._generate_final_analysis(cycle + 1, session_start, cycle_results)
            return False
        except Exception as e:
            print(f"\n💥 Session failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _record_cycle_data(self, cycle: int, result: Dict[str, Any], cycle_start: float):
        """Record comprehensive cycle data for analysis"""
        
        cycle_duration = time.time() - cycle_start
        
        # Record cycle timing
        self.instrumentation.record_cycle_timing(cycle, cycle_duration)
        
        # Record Pathos state if available
        if hasattr(self.agent, 'pathos'):
            pathos_state = self.agent.pathos.current_state
            internal_reward = result.get('internal_reward', 0)
            homeostatic_balance = {'stability': 0.8, 'complexity': 0.6}  # Simplified
            
            self.instrumentation.record_pathos_state(
                cycle, pathos_state, internal_reward, homeostatic_balance
            )
        
        # Record memory events
        if result.get('memory_stored'):
            self.instrumentation.record_memory_event(
                cycle, 'store', salience=0.7, trace_count=cycle + 1
            )
        
        # Record behavior patterns
        tool_used = result.get('tool_used')
        if tool_used:
            self.instrumentation.record_behavior_pattern(
                cycle=cycle,
                intention=result.get('intention', ''),
                tool_used=tool_used,
                reward=result.get('external_reward', 0),
                pattern_similarity=np.random.uniform(0.3, 0.9)
            )
        
        # Record attractor patterns (simplified)
        self.instrumentation.record_attractor_pattern(
            cycle=cycle,
            attractor_count=np.random.randint(1, 5),
            strongest_attractor_strength=np.random.uniform(0.4, 0.9),
            pattern_recognized=np.random.choice([True, False], p=[0.3, 0.7])
        )
    
    def _display_cycle_results(self, cycle: int, result: Dict[str, Any]):
        """Display formatted cycle results"""
        
        intention = result.get('intention', 'No intention generated')
        tool_used = result.get('tool_used', 'None')
        internal_reward = result.get('internal_reward', 0)
        external_reward = result.get('external_reward', 0)
        memory_stored = result.get('memory_stored', False)
        
        print(f"   💭 Intention: {intention[:60]}{'...' if len(intention) > 60 else ''}")
        print(f"   🔧 Tool Used: {tool_used}")
        print(f"   💝 Internal Reward: {internal_reward:.3f}")
        print(f"   🌟 External Reward: {external_reward:.3f}")
        print(f"   💾 Memory Stored: {'✅' if memory_stored else '❌'}")
        
        # Show agent status
        if hasattr(self.agent, 'get_status'):
            status = self.agent.get_status()
            state_norm = status.get('current_state_norm', 0)
            memory_count = status.get('memory_trace_count', 0)
            print(f"   📊 State Norm: {state_norm:.3f} | Memories: {memory_count}")
    
    def _generate_periodic_analysis(self, cycle: int):
        """Generate periodic analysis and visualizations"""
        
        try:
            # Create analysis directory for this cycle
            cycle_dir = self.output_dir / f"cycle_{cycle:03d}"
            cycle_dir.mkdir(exist_ok=True)
            
            # Generate preference drift analysis
            drift_analyzer = PreferenceDriftAnalyzer()
            if self.instrumentation.preference_history:
                print("   📈 Analyzing preference drift...")
                drift_analyzer.visualize_preference_drift(
                    self.instrumentation, 
                    str(cycle_dir / "preference_drift.png")
                )
            
            # Generate attractor pattern analysis
            attractor_detector = AttractorPatternDetector()
            if self.instrumentation.metrics['attractor_patterns']:
                print("   🎯 Analyzing attractor patterns...")
                attractor_detector.visualize_attractor_patterns(
                    self.instrumentation,
                    str(cycle_dir / "attractor_patterns.png")
                )
            
            # Save current Pathos visualization
            if self.pathos_visualizer:
                print("   🧠 Saving Pathos state visualization...")
                self.pathos_visualizer.save_visualization(
                    str(cycle_dir / "pathos_state.png")
                )
            
            print(f"   ✅ Analysis saved to {cycle_dir}")
            
        except Exception as e:
            print(f"   ⚠️ Periodic analysis failed: {e}")
    
    def _generate_final_analysis(self, cycles: int, session_start: float, cycle_results: List):
        """Generate comprehensive final analysis"""
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE FINAL ANALYSIS")
        print("=" * 60)
        
        session_duration = time.time() - session_start
        
        try:
            # Create final analysis directory
            final_dir = self.output_dir / "final_analysis"
            final_dir.mkdir(exist_ok=True)
            
            # Generate comprehensive analysis
            print("🔍 Generating comprehensive analysis...")
            analysis_results = create_comprehensive_analysis(
                collector=self.instrumentation,
                memory_traces=getattr(self.agent.memory, 'traces', []) if hasattr(self.agent, 'memory') else None,
                save_directory=str(final_dir)
            )
            
            # Generate system health dashboard
            print("🏥 Generating system health dashboard...")
            dashboard = SystemHealthDashboard()
            dashboard.generate_dashboard(
                self.instrumentation,
                str(final_dir / "system_health_dashboard.png")
            )
            
            # Save final Pathos visualization
            if self.pathos_visualizer:
                print("🧠 Saving final Pathos visualization...")
                self.pathos_visualizer.save_visualization(
                    str(final_dir / "final_pathos_state.png")
                )
            
            # Generate session summary
            self._generate_session_summary(cycles, session_duration, cycle_results, final_dir)
            
            print(f"\n✅ Complete analysis saved to: {final_dir}")
            print(f"📁 All visualizations available in: {self.output_dir}")
            
            # Display key insights
            if analysis_results:
                self._display_key_insights(analysis_results)
            
        except Exception as e:
            print(f"❌ Final analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_session_summary(self, cycles: int, duration: float, results: List, output_dir: Path):
        """Generate detailed session summary"""
        
        successful_cycles = len([r for r in results if r])
        tools_used = [r.get('tool_used') for r in results if r.get('tool_used')]
        memories_stored = sum(1 for r in results if r.get('memory_stored'))
        
        avg_internal_reward = np.mean([r.get('internal_reward', 0) for r in results if r])
        avg_external_reward = np.mean([r.get('external_reward', 0) for r in results if r])
        
        summary = {
            'session_info': {
                'total_cycles': cycles,
                'successful_cycles': successful_cycles,
                'session_duration_minutes': duration / 60,
                'avg_cycle_time_seconds': duration / max(1, cycles)
            },
            'performance_metrics': {
                'success_rate': successful_cycles / max(1, cycles),
                'tool_usage_count': len(tools_used),
                'unique_tools_used': len(set(tools_used)),
                'memory_storage_rate': memories_stored / max(1, cycles),
                'avg_internal_reward': float(avg_internal_reward),
                'avg_external_reward': float(avg_external_reward)
            },
            'tool_usage_breakdown': dict(zip(*np.unique(tools_used, return_counts=True))) if tools_used else {},
            'instrumentation_summary': self.instrumentation.get_metrics_summary()
        }
        
        # Save summary to JSON
        with open(output_dir / "session_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📋 Session Summary:")
        print(f"   🔄 Cycles: {successful_cycles}/{cycles} successful")
        print(f"   ⏱️ Duration: {duration/60:.1f} minutes")
        print(f"   🔧 Tools Used: {len(tools_used)} times ({len(set(tools_used))} unique)")
        print(f"   💾 Memories: {memories_stored} stored")
        print(f"   💝 Avg Internal Reward: {avg_internal_reward:.3f}")
        print(f"   🌟 Avg External Reward: {avg_external_reward:.3f}")
    
    def _display_key_insights(self, analysis_results: Dict[str, Any]):
        """Display key insights from analysis"""
        
        print(f"\n🔍 KEY INSIGHTS:")
        print("-" * 30)
        
        # Preference insights
        if 'preference_analysis' in analysis_results:
            pref_analysis = analysis_results['preference_analysis']
            if 'summary' in pref_analysis:
                summary = pref_analysis['summary']
                stability = summary.get('average_stability_score', 0)
                drift_pct = summary.get('drift_percentage', 0)
                print(f"🎯 Preferences: {stability:.2f} stability, {drift_pct:.1f}% drift")
        
        # Attractor insights
        if 'attractor_analysis' in analysis_results:
            attr_analysis = analysis_results['attractor_analysis']
            if 'average_attractor_count' in attr_analysis:
                avg_attractors = attr_analysis['average_attractor_count']
                recognition_rate = attr_analysis.get('pattern_recognition_rate', 0)
                print(f"🎯 Attractors: {avg_attractors:.1f} average, {recognition_rate:.1%} recognition")
        
        # System insights
        if 'system_metrics' in analysis_results:
            sys_metrics = analysis_results['system_metrics']
            total_cycles = sys_metrics.get('total_cycles', 0)
            memory_events = sys_metrics.get('total_memory_events', 0)
            print(f"🎯 System: {total_cycles} cycles, {memory_events} memory events")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pathos_visualizer:
            self.pathos_visualizer.close()
        print("🧹 Cleanup complete")


def main():
    """Main function for comprehensive agent demonstration"""
    
    print("🎯 Comprehensive Autonomous Agent with Visualization")
    print("=" * 60)
    
    demo = ComprehensiveAgentDemo()
    
    try:
        # Setup agent
        success = demo.setup_agent_with_monitoring()
        if not success:
            print("❌ Failed to setup agent")
            return
        
        # Interactive menu
        while True:
            print(f"\n🎮 Comprehensive Demo Options:")
            print("1. Run short session (10 cycles)")
            print("2. Run medium session (25 cycles)")
            print("3. Run long session (50 cycles)")
            print("4. Run custom session")
            print("5. Generate analysis from existing data")
            print("6. Exit demo")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                demo.run_comprehensive_session(10, 3)
            elif choice == "2":
                demo.run_comprehensive_session(25, 5)
            elif choice == "3":
                demo.run_comprehensive_session(50, 10)
            elif choice == "4":
                try:
                    cycles = int(input("Enter number of cycles: "))
                    interval = int(input("Enter visualization interval: "))
                    demo.run_comprehensive_session(cycles, interval)
                except ValueError:
                    print("❌ Invalid input")
            elif choice == "5":
                if demo.instrumentation:
                    print("📊 Generating analysis from current data...")
                    create_comprehensive_analysis(
                        demo.instrumentation,
                        save_directory=str(demo.output_dir / "manual_analysis")
                    )
                else:
                    print("❌ No data available")
            elif choice == "6":
                break
            else:
                print("❌ Invalid choice")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()