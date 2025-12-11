"""
Visualization and analysis tools for the Autonomous Logos-Pathos-Memory Agent.

This module provides comprehensive visualization and analysis capabilities for:
- Real-time Pathos state visualization
- Preference drift analysis and attractor pattern detection
- Memory network visualization and system health dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, deque
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from .logging_config import InstrumentationCollector
from .models import MemoryTrace


class PathosStateVisualizer:
    """Real-time visualization of Pathos affective state dynamics"""
    
    def __init__(self, state_dimension: int = 128, history_length: int = 100):
        self.state_dimension = state_dimension
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        self.time_history = deque(maxlen=history_length)
        
        # Setup matplotlib for real-time plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Pathos State Real-Time Visualization')
        
        # Initialize plots
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup the visualization plots"""
        # State norm over time
        self.axes[0, 0].set_title('State Vector Norm Over Time')
        self.axes[0, 0].set_xlabel('Time Steps')
        self.axes[0, 0].set_ylabel('||F(t)||')
        self.state_norm_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2)
        
        # Internal reward over time
        self.axes[0, 1].set_title('Internal Reward Over Time')
        self.axes[0, 1].set_xlabel('Time Steps')
        self.axes[0, 1].set_ylabel('Internal Reward')
        self.reward_line, = self.axes[0, 1].plot([], [], 'r-', linewidth=2)
        
        # State vector heatmap (2D projection)
        self.axes[1, 0].set_title('State Vector Components (PCA Projection)')
        self.state_heatmap = None
        
        # Phase space plot (state norm vs reward)
        self.axes[1, 1].set_title('Phase Space (State Norm vs Reward)')
        self.axes[1, 1].set_xlabel('State Norm')
        self.axes[1, 1].set_ylabel('Internal Reward')
        self.phase_scatter = self.axes[1, 1].scatter([], [], c=[], cmap='viridis', alpha=0.6)
        
        plt.tight_layout()
    
    def update_state(self, state_vector: np.ndarray, internal_reward: float, timestamp: Optional[str] = None):
        """Update visualization with new state data"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Store data
        self.state_history.append(state_vector.copy())
        self.reward_history.append(internal_reward)
        self.time_history.append(len(self.time_history))
        
        # Update plots
        self._update_plots()
    
    def _update_plots(self):
        """Update all visualization plots"""
        if len(self.state_history) < 2:
            return
        
        # Calculate state norms
        state_norms = [np.linalg.norm(state) for state in self.state_history]
        
        # Update state norm plot
        self.state_norm_line.set_data(list(self.time_history), state_norms)
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()
        
        # Update reward plot
        self.reward_line.set_data(list(self.time_history), list(self.reward_history))
        self.axes[0, 1].relim()
        self.axes[0, 1].autoscale_view()
        
        # Update state heatmap (PCA projection)
        if len(self.state_history) >= 2:
            states_matrix = np.array(list(self.state_history))
            if states_matrix.shape[1] > 2:
                pca = PCA(n_components=min(10, states_matrix.shape[1]))
                states_pca = pca.fit_transform(states_matrix)
                
                self.axes[1, 0].clear()
                self.axes[1, 0].set_title('State Vector Components (PCA Projection)')
                im = self.axes[1, 0].imshow(states_pca[-20:].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                self.axes[1, 0].set_xlabel('Time Steps (Recent 20)')
                self.axes[1, 0].set_ylabel('PCA Components')
        
        # Update phase space plot
        if len(state_norms) >= 2:
            colors = np.arange(len(state_norms))
            self.axes[1, 1].clear()
            self.axes[1, 1].set_title('Phase Space (State Norm vs Reward)')
            self.axes[1, 1].set_xlabel('State Norm')
            self.axes[1, 1].set_ylabel('Internal Reward')
            scatter = self.axes[1, 1].scatter(state_norms, list(self.reward_history), 
                                            c=colors, cmap='viridis', alpha=0.6)
            
            # Add trajectory line
            self.axes[1, 1].plot(state_norms, list(self.reward_history), 'k-', alpha=0.3, linewidth=1)
        
        plt.draw()
        plt.pause(0.01)
    
    def save_visualization(self, filepath: str):
        """Save current visualization to file"""
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)


class PreferenceDriftAnalyzer:
    """Analyzer for preference drift patterns and trends"""
    
    def __init__(self):
        self.preference_data = defaultdict(list)
        self.drift_threshold = 0.3
    
    def analyze_drift_from_collector(self, collector: InstrumentationCollector) -> Dict[str, Any]:
        """Analyze preference drift from instrumentation collector"""
        if not collector.preference_history:
            return {'error': 'No preference data available'}
        
        analysis = {}
        
        for category, history in collector.preference_history.items():
            if len(history) < 3:
                continue
            
            timestamps, weights = zip(*history)
            weights = np.array(weights)
            
            # Calculate drift metrics
            drift_analysis = self._analyze_category_drift(category, weights, timestamps)
            analysis[category] = drift_analysis
        
        # Overall drift summary
        analysis['summary'] = self._calculate_overall_drift_summary(analysis)
        
        return analysis
    
    def _analyze_category_drift(self, category: str, weights: np.ndarray, timestamps: List[str]) -> Dict[str, Any]:
        """Analyze drift for a specific category"""
        # Basic statistics
        mean_weight = float(np.mean(weights))
        std_weight = float(np.std(weights))
        min_weight = float(np.min(weights))
        max_weight = float(np.max(weights))
        
        # Trend analysis (linear regression slope)
        x = np.arange(len(weights))
        slope = np.polyfit(x, weights, 1)[0] if len(weights) > 1 else 0.0
        
        # Volatility (rolling standard deviation)
        window_size = min(5, len(weights))
        if window_size > 1:
            rolling_std = pd.Series(weights).rolling(window=window_size).std().fillna(0)
            avg_volatility = float(np.mean(rolling_std))
        else:
            avg_volatility = 0.0
        
        # Drift detection
        total_drift = max_weight - min_weight
        significant_drift = total_drift > self.drift_threshold
        
        # Change points detection (simple threshold-based)
        changes = np.abs(np.diff(weights))
        major_changes = np.where(changes > 0.2)[0]
        
        return {
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'total_drift': total_drift,
            'trend_slope': float(slope),
            'avg_volatility': avg_volatility,
            'significant_drift': significant_drift,
            'major_change_points': len(major_changes),
            'stability_score': 1.0 - (avg_volatility + abs(slope)) / 2.0
        }
    
    def _calculate_overall_drift_summary(self, category_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall drift summary across all categories"""
        if not category_analyses:
            return {}
        
        # Filter out summary key if it exists
        analyses = {k: v for k, v in category_analyses.items() if k != 'summary'}
        
        if not analyses:
            return {}
        
        total_categories = len(analyses)
        significant_drifts = sum(1 for analysis in analyses.values() if analysis.get('significant_drift', False))
        avg_stability = np.mean([analysis.get('stability_score', 0) for analysis in analyses.values()])
        most_volatile = max(analyses.items(), key=lambda x: x[1].get('avg_volatility', 0))
        
        return {
            'total_categories': total_categories,
            'categories_with_significant_drift': significant_drifts,
            'drift_percentage': (significant_drifts / total_categories) * 100,
            'average_stability_score': float(avg_stability),
            'most_volatile_category': most_volatile[0],
            'most_volatile_score': most_volatile[1].get('avg_volatility', 0)
        }
    
    def visualize_preference_drift(self, collector: InstrumentationCollector, save_path: Optional[str] = None):
        """Create visualization of preference drift patterns"""
        if not collector.preference_history:
            print("No preference data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Preference Drift Analysis')
        
        # Plot 1: Preference weights over time
        ax1 = axes[0, 0]
        ax1.set_title('Preference Weights Over Time')
        ax1.set_xlabel('Update Number')
        ax1.set_ylabel('Weight Value')
        
        for category, history in collector.preference_history.items():
            if len(history) > 1:
                _, weights = zip(*history)
                ax1.plot(range(len(weights)), weights, marker='o', label=category, linewidth=2)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drift magnitude distribution
        ax2 = axes[0, 1]
        ax2.set_title('Drift Magnitude Distribution')
        
        drift_magnitudes = []
        for category, history in collector.preference_history.items():
            if len(history) > 1:
                _, weights = zip(*history)
                weights = np.array(weights)
                drift_magnitudes.append(np.max(weights) - np.min(weights))
        
        if drift_magnitudes:
            ax2.hist(drift_magnitudes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(self.drift_threshold, color='red', linestyle='--', label=f'Threshold ({self.drift_threshold})')
            ax2.legend()
        
        ax2.set_xlabel('Total Drift Magnitude')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Volatility analysis
        ax3 = axes[1, 0]
        ax3.set_title('Preference Volatility by Category')
        
        categories = []
        volatilities = []
        
        for category, history in collector.preference_history.items():
            if len(history) > 2:
                _, weights = zip(*history)
                weights = np.array(weights)
                changes = np.abs(np.diff(weights))
                volatility = np.std(changes)
                categories.append(category)
                volatilities.append(volatility)
        
        if categories:
            bars = ax3.bar(categories, volatilities, color='lightcoral', alpha=0.7)
            ax3.set_ylabel('Volatility (Std of Changes)')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Stability scores
        ax4 = axes[1, 1]
        ax4.set_title('Category Stability Scores')
        
        analysis = self.analyze_drift_from_collector(collector)
        if 'summary' not in analysis:
            stability_categories = []
            stability_scores = []
            
            for category, cat_analysis in analysis.items():
                if isinstance(cat_analysis, dict) and 'stability_score' in cat_analysis:
                    stability_categories.append(category)
                    stability_scores.append(cat_analysis['stability_score'])
            
            if stability_categories:
                colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' for score in stability_scores]
                bars = ax4.bar(stability_categories, stability_scores, color=colors, alpha=0.7)
                ax4.set_ylabel('Stability Score')
                ax4.set_ylim(0, 1)
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AttractorPatternDetector:
    """Detector and visualizer for attractor patterns in agent behavior"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.detected_patterns = []
    
    def detect_patterns_from_collector(self, collector: InstrumentationCollector) -> Dict[str, Any]:
        """Detect attractor patterns from instrumentation data"""
        if not collector.metrics['attractor_patterns']:
            return {'error': 'No attractor pattern data available'}
        
        patterns = collector.metrics['attractor_patterns']
        
        # Analyze attractor evolution
        attractor_counts = [p['attractor_count'] for p in patterns]
        attractor_strengths = [p['strongest_attractor_strength'] for p in patterns]
        pattern_recognitions = [p['pattern_recognized'] for p in patterns]
        
        # Pattern statistics
        analysis = {
            'total_cycles_analyzed': len(patterns),
            'average_attractor_count': float(np.mean(attractor_counts)),
            'max_attractor_count': int(np.max(attractor_counts)),
            'average_attractor_strength': float(np.mean(attractor_strengths)),
            'max_attractor_strength': float(np.max(attractor_strengths)),
            'pattern_recognition_rate': float(np.mean(pattern_recognitions)),
            'attractor_growth_trend': self._calculate_trend(attractor_counts),
            'strength_trend': self._calculate_trend(attractor_strengths)
        }
        
        # Detect stable periods
        stable_periods = self._detect_stable_periods(attractor_counts, attractor_strengths)
        analysis['stable_periods'] = stable_periods
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend (slope) of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _detect_stable_periods(self, counts: List[int], strengths: List[float]) -> List[Dict[str, Any]]:
        """Detect periods of stable attractor behavior"""
        if len(counts) < 5:
            return []
        
        stable_periods = []
        window_size = 5
        stability_threshold = 0.1  # Low variance indicates stability
        
        for i in range(len(counts) - window_size + 1):
            count_window = counts[i:i + window_size]
            strength_window = strengths[i:i + window_size]
            
            count_variance = np.var(count_window)
            strength_variance = np.var(strength_window)
            
            if count_variance < stability_threshold and strength_variance < stability_threshold:
                stable_periods.append({
                    'start_cycle': i,
                    'end_cycle': i + window_size - 1,
                    'duration': window_size,
                    'avg_attractor_count': float(np.mean(count_window)),
                    'avg_strength': float(np.mean(strength_window)),
                    'stability_score': 1.0 - (count_variance + strength_variance)
                })
        
        return stable_periods
    
    def visualize_attractor_patterns(self, collector: InstrumentationCollector, save_path: Optional[str] = None):
        """Visualize attractor patterns and dynamics"""
        if not collector.metrics['attractor_patterns']:
            print("No attractor pattern data available for visualization")
            return
        
        patterns = collector.metrics['attractor_patterns']
        cycles = [p['cycle'] for p in patterns]
        attractor_counts = [p['attractor_count'] for p in patterns]
        attractor_strengths = [p['strongest_attractor_strength'] for p in patterns]
        pattern_recognitions = [p['pattern_recognized'] for p in patterns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Attractor Pattern Analysis')
        
        # Plot 1: Attractor count over time
        ax1 = axes[0, 0]
        ax1.plot(cycles, attractor_counts, 'b-o', linewidth=2, markersize=4)
        ax1.set_title('Attractor Count Over Time')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('Number of Attractors')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attractor strength over time
        ax2 = axes[0, 1]
        ax2.plot(cycles, attractor_strengths, 'r-o', linewidth=2, markersize=4)
        ax2.set_title('Strongest Attractor Strength Over Time')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Attractor Strength')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Pattern recognition events
        ax3 = axes[1, 0]
        recognition_cycles = [c for c, r in zip(cycles, pattern_recognitions) if r]
        if recognition_cycles:
            ax3.scatter(recognition_cycles, [1] * len(recognition_cycles), 
                       c='green', s=50, alpha=0.7, label='Pattern Recognized')
        
        no_recognition_cycles = [c for c, r in zip(cycles, pattern_recognitions) if not r]
        if no_recognition_cycles:
            ax3.scatter(no_recognition_cycles, [0] * len(no_recognition_cycles), 
                       c='red', s=50, alpha=0.7, label='No Pattern')
        
        ax3.set_title('Pattern Recognition Events')
        ax3.set_xlabel('Cycle')
        ax3.set_ylabel('Pattern Recognized')
        ax3.set_ylim(-0.1, 1.1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Attractor dynamics phase space
        ax4 = axes[1, 1]
        scatter = ax4.scatter(attractor_counts, attractor_strengths, 
                            c=cycles, cmap='viridis', alpha=0.7, s=50)
        ax4.set_title('Attractor Dynamics Phase Space')
        ax4.set_xlabel('Attractor Count')
        ax4.set_ylabel('Strongest Attractor Strength')
        plt.colorbar(scatter, ax=ax4, label='Cycle')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class MemoryNetworkVisualizer:
    """Visualizer for memory network structure and relationships"""
    
    def __init__(self):
        self.similarity_threshold = 0.7
    
    def visualize_memory_network(self, memory_traces: List[MemoryTrace], 
                                max_nodes: int = 50, save_path: Optional[str] = None):
        """Create network visualization of memory relationships"""
        if len(memory_traces) < 2:
            print("Need at least 2 memory traces for network visualization")
            return
        
        # Limit number of traces for visualization performance
        traces = memory_traces[-max_nodes:] if len(memory_traces) > max_nodes else memory_traces
        
        # Create similarity matrix
        similarity_matrix = self._compute_similarity_matrix(traces)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, trace in enumerate(traces):
            G.add_node(i, 
                      reward=trace.reward,
                      salience=trace.salience,
                      timestamp=trace.timestamp,
                      category=trace.semantic_vector.semantic_category)
        
        # Add edges based on similarity
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                similarity = similarity_matrix[i, j]
                if similarity > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Network plot
        ax1 = axes[0]
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Node colors based on reward
        node_colors = [trace.reward for trace in traces]
        node_sizes = [trace.salience * 500 + 100 for trace in traces]  # Size based on salience
        
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                              node_size=node_sizes, cmap='RdYlBu', alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.5, width=0.5)
        
        ax1.set_title('Memory Network Structure')
        ax1.set_xlabel('Node color: Reward | Node size: Salience')
        ax1.axis('off')
        
        # Memory statistics
        ax2 = axes[1]
        
        # Plot memory distribution over time
        timestamps = [trace.timestamp for trace in traces]
        rewards = [trace.reward for trace in traces]
        saliences = [trace.salience for trace in traces]
        
        # Normalize timestamps for plotting
        if timestamps:
            min_time = min(timestamps)
            normalized_times = [(t - min_time) for t in timestamps]
            
            scatter = ax2.scatter(normalized_times, rewards, c=saliences, 
                                cmap='viridis', alpha=0.7, s=60)
            ax2.set_xlabel('Time (relative)')
            ax2.set_ylabel('Reward')
            ax2.set_title('Memory Traces: Reward vs Time')
            plt.colorbar(scatter, ax=ax2, label='Salience')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print network statistics
        print(f"Memory Network Statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Density: {nx.density(G):.3f}")
        print(f"  Connected components: {nx.number_connected_components(G)}")
        
        if G.number_of_edges() > 0:
            print(f"  Average clustering: {nx.average_clustering(G):.3f}")
    
    def _compute_similarity_matrix(self, traces: List[MemoryTrace]) -> np.ndarray:
        """Compute similarity matrix between memory traces"""
        n = len(traces)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Combine affective and semantic similarity
                    affect_sim = self._cosine_similarity(traces[i].affect_state, traces[j].affect_state)
                    semantic_sim = self._cosine_similarity(
                        traces[i].semantic_vector.embedding, 
                        traces[j].semantic_vector.embedding
                    )
                    
                    # Weighted combination
                    combined_sim = 0.6 * affect_sim + 0.4 * semantic_sim
                    similarity_matrix[i, j] = combined_sim
                    similarity_matrix[j, i] = combined_sim
        
        return similarity_matrix
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class SystemHealthDashboard:
    """Comprehensive system health monitoring dashboard"""
    
    def __init__(self):
        self.health_metrics = {}
    
    def generate_dashboard(self, collector: InstrumentationCollector, save_path: Optional[str] = None):
        """Generate comprehensive system health dashboard"""
        # Collect all metrics
        metrics_summary = collector.get_metrics_summary()
        preference_summary = collector.get_preference_drift_summary()
        behavior_summary = collector.get_behavior_pattern_summary()
        
        # Create dashboard
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Autonomous Agent System Health Dashboard', fontsize=16, fontweight='bold')
        
        # 1. System Overview (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_system_overview(ax1, metrics_summary)
        
        # 2. Performance Metrics (top-center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_performance_metrics(ax2, metrics_summary, collector)
        
        # 3. Preference Health (top-center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_preference_health(ax3, preference_summary)
        
        # 4. Behavior Stability (top-right)
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_behavior_stability(ax4, behavior_summary)
        
        # 5. Memory System Health (middle-left)
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_memory_health(ax5, collector)
        
        # 6. Tool Usage Analysis (middle-center-left)
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_tool_usage(ax6, collector)
        
        # 7. Error Rate Analysis (middle-center-right)
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_error_analysis(ax7, collector)
        
        # 8. Cycle Performance (middle-right)
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_cycle_performance(ax8, collector)
        
        # 9. Health Score Summary (bottom, spanning 2 columns)
        ax9 = fig.add_subplot(gs[2, 0:2])
        self._plot_health_scores(ax9, metrics_summary, preference_summary, behavior_summary)
        
        # 10. System Alerts (bottom-right, spanning 2 columns)
        ax10 = fig.add_subplot(gs[2, 2:4])
        self._plot_system_alerts(ax10, collector, preference_summary, behavior_summary)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_system_overview(self, ax, metrics_summary):
        """Plot system overview metrics"""
        ax.set_title('System Overview', fontweight='bold')
        
        # Create text summary
        runtime_hours = metrics_summary.get('runtime_seconds', 0) / 3600
        total_cycles = metrics_summary.get('total_cycles', 0)
        
        overview_text = f"""
Runtime: {runtime_hours:.1f} hours
Total Cycles: {total_cycles}
Memory Events: {metrics_summary.get('total_memory_events', 0)}
Tool Calls: {metrics_summary.get('total_tool_calls', 0)}
Errors: {metrics_summary.get('total_errors', 0)}
        """.strip()
        
        ax.text(0.1, 0.5, overview_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.axis('off')
    
    def _plot_performance_metrics(self, ax, metrics_summary, collector):
        """Plot performance metrics"""
        ax.set_title('Performance Metrics', fontweight='bold')
        
        avg_cycle_time = metrics_summary.get('avg_cycle_time', 0)
        
        # Create gauge-like visualization
        if avg_cycle_time > 0:
            # Performance categories
            categories = ['Cycle Time', 'Memory Efficiency', 'Tool Success Rate']
            
            # Calculate scores (normalized to 0-1)
            cycle_score = max(0, 1 - (avg_cycle_time / 5.0))  # Assume 5s is poor performance
            
            # Memory efficiency (based on memory events vs cycles)
            total_cycles = max(1, metrics_summary.get('total_cycles', 1))
            memory_events = metrics_summary.get('total_memory_events', 0)
            memory_score = min(1.0, memory_events / total_cycles)
            
            # Tool success rate (simplified)
            tool_score = 0.8  # Default assumption
            
            scores = [cycle_score, memory_score, tool_score]
            colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in scores]
            
            bars = ax.barh(categories, scores, color=colors, alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Performance Score')
            
            # Add score labels
            for bar, score in zip(bars, scores):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{score:.2f}', va='center')
        else:
            ax.text(0.5, 0.5, 'No performance data', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.axis('off')
    
    def _plot_preference_health(self, ax, preference_summary):
        """Plot preference system health"""
        ax.set_title('Preference System Health', fontweight='bold')
        
        if preference_summary and preference_summary.get('total_categories_tracked', 0) > 0:
            # Create pie chart of drift status
            categories_tracked = preference_summary.get('total_categories_tracked', 0)
            categories_with_drift = preference_summary.get('categories_with_significant_drift', 0)
            stable_categories = categories_tracked - categories_with_drift
            
            if categories_tracked > 0:
                sizes = [stable_categories, categories_with_drift]
                labels = ['Stable', 'Drifting']
                colors = ['lightgreen', 'lightcoral']
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
                ax.set_title('Preference Stability')
            else:
                ax.text(0.5, 0.5, 'No preference data', transform=ax.transAxes, 
                       ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'No preference data', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.axis('off')
    
    def _plot_behavior_stability(self, ax, behavior_summary):
        """Plot behavior stability metrics"""
        ax.set_title('Behavior Stability', fontweight='bold')
        
        if behavior_summary:
            stability = behavior_summary.get('behavior_stability', 0)
            exploration = behavior_summary.get('exploration_level', 0)
            
            # Create radar-like plot
            metrics = ['Stability', 'Exploration', 'Adaptability']
            values = [stability, exploration, (stability + exploration) / 2]  # Adaptability as combination
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
            ax.fill(angles, values, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No behavior data', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.axis('off')
    
    def _plot_memory_health(self, ax, collector):
        """Plot memory system health"""
        ax.set_title('Memory System Health', fontweight='bold')
        
        memory_events = collector.metrics.get('memory_events', [])
        
        if memory_events:
            # Analyze memory storage rate over time
            recent_events = memory_events[-20:]  # Last 20 events
            storage_events = [e for e in recent_events if e['event_type'] == 'store']
            
            if storage_events:
                saliences = [e['salience'] for e in storage_events]
                
                # Plot salience distribution
                ax.hist(saliences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(saliences), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(saliences):.2f}')
                ax.set_xlabel('Salience Score')
                ax.set_ylabel('Frequency')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No storage events', transform=ax.transAxes, 
                       ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'No memory data', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.axis('off')
    
    def _plot_tool_usage(self, ax, collector):
        """Plot tool usage analysis"""
        ax.set_title('Tool Usage Analysis', fontweight='bold')
        
        tool_usage = collector.metrics.get('tool_usage', [])
        
        if tool_usage:
            # Analyze tool success rates
            tool_stats = defaultdict(lambda: {'success': 0, 'total': 0})
            
            for usage in tool_usage:
                tool_name = usage['tool_name']
                tool_stats[tool_name]['total'] += 1
                if usage['success']:
                    tool_stats[tool_name]['success'] += 1
            
            if tool_stats:
                tools = list(tool_stats.keys())
                success_rates = [tool_stats[tool]['success'] / tool_stats[tool]['total'] 
                               for tool in tools]
                
                colors = ['green' if rate > 0.7 else 'orange' if rate > 0.4 else 'red' 
                         for rate in success_rates]
                
                bars = ax.bar(tools, success_rates, color=colors, alpha=0.7)
                ax.set_ylabel('Success Rate')
                ax.set_ylim(0, 1)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for bar, rate in zip(bars, success_rates):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{rate:.2f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No tool statistics', transform=ax.transAxes, 
                       ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'No tool usage data', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.axis('off')
    
    def _plot_error_analysis(self, ax, collector):
        """Plot error analysis"""
        ax.set_title('Error Analysis', fontweight='bold')
        
        errors = collector.metrics.get('error_events', [])
        
        if errors:
            # Analyze error types
            error_types = defaultdict(int)
            for error in errors:
                error_types[error['error_type']] += 1
            
            if error_types:
                types = list(error_types.keys())
                counts = list(error_types.values())
                
                ax.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
            else:
                ax.text(0.5, 0.5, 'No error data', transform=ax.transAxes, 
                       ha='center', va='center')
        else:
            # No errors is good!
            ax.text(0.5, 0.5, '✓ No Errors Detected', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, color='green', fontweight='bold')
            ax.axis('off')
    
    def _plot_cycle_performance(self, ax, collector):
        """Plot cycle performance over time"""
        ax.set_title('Cycle Performance', fontweight='bold')
        
        cycle_timings = collector.metrics.get('cycle_timings', [])
        
        if cycle_timings:
            cycles = [t['cycle'] for t in cycle_timings]
            durations = [t['total_duration'] for t in cycle_timings]
            
            ax.plot(cycles, durations, 'b-', linewidth=2, alpha=0.7)
            ax.set_xlabel('Cycle Number')
            ax.set_ylabel('Duration (seconds)')
            
            # Add trend line
            if len(cycles) > 1:
                z = np.polyfit(cycles, durations, 1)
                p = np.poly1d(z)
                ax.plot(cycles, p(cycles), 'r--', alpha=0.8, 
                       label=f'Trend: {z[0]:.4f}s/cycle')
                ax.legend()
            
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No cycle timing data', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.axis('off')
    
    def _plot_health_scores(self, ax, metrics_summary, preference_summary, behavior_summary):
        """Plot overall health scores"""
        ax.set_title('System Health Scores', fontweight='bold')
        
        # Calculate health scores
        scores = {}
        
        # Performance score
        avg_cycle_time = metrics_summary.get('avg_cycle_time', 0)
        scores['Performance'] = max(0, min(1, 1 - (avg_cycle_time / 5.0))) if avg_cycle_time > 0 else 0.8
        
        # Preference stability score
        if preference_summary and preference_summary.get('total_categories_tracked', 0) > 0:
            stability_score = preference_summary.get('average_stability_score', 0.5)
            scores['Preferences'] = stability_score
        else:
            scores['Preferences'] = 0.5
        
        # Behavior score
        if behavior_summary:
            behavior_score = behavior_summary.get('behavior_stability', 0.5)
            scores['Behavior'] = behavior_score
        else:
            scores['Behavior'] = 0.5
        
        # Memory score (based on activity)
        memory_events = metrics_summary.get('total_memory_events', 0)
        total_cycles = max(1, metrics_summary.get('total_cycles', 1))
        memory_activity = min(1.0, memory_events / total_cycles)
        scores['Memory'] = memory_activity
        
        # Error score (inverse of error rate)
        total_errors = metrics_summary.get('total_errors', 0)
        error_rate = total_errors / max(1, total_cycles)
        scores['Reliability'] = max(0, 1 - error_rate)
        
        # Create horizontal bar chart
        categories = list(scores.keys())
        values = list(scores.values())
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
        
        bars = ax.barh(categories, values, color=colors, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Health Score')
        
        # Add score labels
        for bar, score in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{score:.2f}', va='center')
        
        # Add overall health score
        overall_score = np.mean(values)
        ax.text(0.5, -0.15, f'Overall Health: {overall_score:.2f}', 
               transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
               color='green' if overall_score > 0.7 else 'orange' if overall_score > 0.4 else 'red')
    
    def _plot_system_alerts(self, ax, collector, preference_summary, behavior_summary):
        """Plot system alerts and warnings"""
        ax.set_title('System Alerts', fontweight='bold')
        
        alerts = []
        
        # Check for various alert conditions
        
        # Error rate alerts
        total_errors = len(collector.metrics.get('error_events', []))
        total_cycles = len(collector.metrics.get('cycle_timings', []))
        if total_cycles > 0:
            error_rate = total_errors / total_cycles
            if error_rate > 0.1:
                alerts.append(f"⚠️ High error rate: {error_rate:.1%}")
        
        # Preference drift alerts
        if preference_summary:
            drift_percentage = preference_summary.get('drift_percentage', 0)
            if drift_percentage > 50:
                alerts.append(f"⚠️ High preference drift: {drift_percentage:.1f}%")
        
        # Behavior stability alerts
        if behavior_summary:
            stability = behavior_summary.get('behavior_stability', 1.0)
            if stability < 0.3:
                alerts.append(f"⚠️ Low behavior stability: {stability:.2f}")
        
        # Memory system alerts
        memory_events = collector.metrics.get('memory_events', [])
        recent_memory_events = memory_events[-10:] if len(memory_events) > 10 else memory_events
        if len(recent_memory_events) == 0 and total_cycles > 5:
            alerts.append("⚠️ No recent memory activity")
        
        # Tool usage alerts
        tool_usage = collector.metrics.get('tool_usage', [])
        if tool_usage:
            recent_tools = tool_usage[-10:]
            failed_tools = [t for t in recent_tools if not t['success']]
            if len(failed_tools) / len(recent_tools) > 0.5:
                alerts.append(f"⚠️ High tool failure rate: {len(failed_tools)}/{len(recent_tools)}")
        
        # Display alerts
        if alerts:
            alert_text = '\n'.join(alerts)
            ax.text(0.05, 0.95, alert_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="lightyellow", edgecolor="orange"))
        else:
            ax.text(0.5, 0.5, '✓ All Systems Normal', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, color='green', fontweight='bold')
        
        ax.axis('off')


# Convenience function to create all visualizations
def create_comprehensive_analysis(collector: InstrumentationCollector, 
                                memory_traces: Optional[List[MemoryTrace]] = None,
                                save_directory: Optional[str] = None):
    """
    Create comprehensive analysis and visualization of agent performance.
    
    Args:
        collector: InstrumentationCollector with monitoring data
        memory_traces: Optional list of memory traces for network visualization
        save_directory: Optional directory to save all visualizations
    """
    print("Creating comprehensive agent analysis...")
    
    # Create analyzers and visualizers
    drift_analyzer = PreferenceDriftAnalyzer()
    attractor_detector = AttractorPatternDetector()
    memory_visualizer = MemoryNetworkVisualizer()
    dashboard = SystemHealthDashboard()
    
    # Generate all visualizations
    try:
        # 1. System Health Dashboard
        print("Generating system health dashboard...")
        dashboard_path = f"{save_directory}/system_health_dashboard.png" if save_directory else None
        dashboard.generate_dashboard(collector, dashboard_path)
        
        # 2. Preference Drift Analysis
        print("Analyzing preference drift...")
        drift_path = f"{save_directory}/preference_drift_analysis.png" if save_directory else None
        drift_analyzer.visualize_preference_drift(collector, drift_path)
        
        # 3. Attractor Pattern Analysis
        print("Analyzing attractor patterns...")
        attractor_path = f"{save_directory}/attractor_patterns.png" if save_directory else None
        attractor_detector.visualize_attractor_patterns(collector, attractor_path)
        
        # 4. Memory Network Visualization (if traces provided)
        if memory_traces:
            print("Visualizing memory network...")
            memory_path = f"{save_directory}/memory_network.png" if save_directory else None
            memory_visualizer.visualize_memory_network(memory_traces, save_path=memory_path)
        
        print("Comprehensive analysis complete!")
        
        # Return analysis summaries
        return {
            'preference_analysis': drift_analyzer.analyze_drift_from_collector(collector),
            'attractor_analysis': attractor_detector.detect_patterns_from_collector(collector),
            'system_metrics': collector.get_metrics_summary()
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None