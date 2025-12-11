"""
Attractor Patterns Tab

Behavioral attractor pattern analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict, deque

from .base_tab import BaseTab


class AttractorTab(BaseTab):
    """Attractor pattern analysis tab."""
    
    def __init__(self, notebook, collector):
        """Initialize attractor tab."""
        self.collector = collector
        self.attractor_threshold = 0.1
        self.min_pattern_length = 5
        super().__init__(notebook, "Attractor Patterns", "🌀")
        self._create_attractor_display()
    
    def _create_attractor_display(self):
        """Create the attractor pattern analysis display."""
        # Create figure with subplots
        fig = Figure(figsize=(16, 12), facecolor='white')
        fig.suptitle('Attractor Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots in a 3x3 grid
        self.ax_pattern_count = fig.add_subplot(3, 3, 1)
        self.ax_strength_evolution = fig.add_subplot(3, 3, 2)
        self.ax_pattern_stability = fig.add_subplot(3, 3, 3)
        self.ax_behavior_clusters = fig.add_subplot(3, 3, 4)
        self.ax_intention_patterns = fig.add_subplot(3, 3, 5)
        self.ax_reward_attractors = fig.add_subplot(3, 3, 6)
        self.ax_phase_space = fig.add_subplot(3, 3, 7)
        self.ax_pattern_transitions = fig.add_subplot(3, 3, 8)
        self.ax_summary_stats = fig.add_subplot(3, 3, 9)
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def update_display(self):
        """Update attractor display with current data."""
        try:
            # Clear all axes
            for ax in [self.ax_pattern_count, self.ax_strength_evolution, self.ax_pattern_stability,
                      self.ax_behavior_clusters, self.ax_intention_patterns, self.ax_reward_attractors,
                      self.ax_phase_space, self.ax_pattern_transitions, self.ax_summary_stats]:
                ax.clear()
            
            # Check if we have attractor data
            if not hasattr(self.collector, 'metrics') or 'attractor_patterns' not in self.collector.metrics:
                self._show_no_data_message()
                return
            
            attractor_data = self.collector.metrics['attractor_patterns']
            behavior_data = self.collector.metrics.get('behavior_patterns', [])
            
            if not attractor_data:
                self._show_no_data_message()
                return
            
            # Plot attractor analysis
            self._plot_pattern_count_evolution(attractor_data)
            self._plot_strength_evolution(attractor_data)
            self._plot_pattern_stability(attractor_data)
            self._plot_behavior_clusters(behavior_data)
            self._plot_intention_patterns(behavior_data)
            self._plot_reward_attractors(attractor_data, behavior_data)
            self._plot_phase_space(attractor_data)
            self._plot_pattern_transitions(behavior_data)
            self._plot_summary_statistics(attractor_data, behavior_data)
            
            # Refresh canvas
            if self.canvas:
                self.canvas.draw()
                
        except Exception as e:
            self._show_error_message(f"Error updating attractor display: {str(e)}")
    
    def _plot_pattern_count_evolution(self, attractor_data):
        """Plot evolution of attractor pattern count over time."""
        self.ax_pattern_count.set_title('Attractor Count Evolution')
        self.ax_pattern_count.set_xlabel('Cycle')
        self.ax_pattern_count.set_ylabel('Number of Attractors')
        
        cycles = [entry['cycle'] for entry in attractor_data]
        counts = [entry['attractor_count'] for entry in attractor_data]
        
        self.ax_pattern_count.plot(cycles, counts, 'b-', linewidth=2, marker='o', markersize=4)
        self.ax_pattern_count.fill_between(cycles, counts, alpha=0.3)
        self.ax_pattern_count.grid(True, alpha=0.3)
        
        # Add average line
        if counts:
            avg_count = np.mean(counts)
            self.ax_pattern_count.axhline(y=avg_count, color='red', linestyle='--', 
                                        label=f'Average: {avg_count:.1f}')
            self.ax_pattern_count.legend()
    
    def _plot_strength_evolution(self, attractor_data):
        """Plot evolution of strongest attractor strength."""
        self.ax_strength_evolution.set_title('Strongest Attractor Strength')
        self.ax_strength_evolution.set_xlabel('Cycle')
        self.ax_strength_evolution.set_ylabel('Strength')
        
        cycles = [entry['cycle'] for entry in attractor_data]
        strengths = [entry['strongest_attractor_strength'] for entry in attractor_data]
        
        # Color code by strength level
        colors = ['red' if s > 0.8 else 'orange' if s > 0.5 else 'green' for s in strengths]
        
        self.ax_strength_evolution.scatter(cycles, strengths, c=colors, alpha=0.7, s=30)
        self.ax_strength_evolution.plot(cycles, strengths, 'k-', alpha=0.3, linewidth=1)
        self.ax_strength_evolution.grid(True, alpha=0.3)
        
        # Add threshold lines
        self.ax_strength_evolution.axhline(y=0.5, color='orange', linestyle=':', label='Medium Threshold')
        self.ax_strength_evolution.axhline(y=0.8, color='red', linestyle=':', label='High Threshold')
        self.ax_strength_evolution.legend()
    
    def _plot_pattern_stability(self, attractor_data):
        """Plot pattern recognition stability."""
        self.ax_pattern_stability.set_title('Pattern Recognition Rate')
        self.ax_pattern_stability.set_xlabel('Cycle Window')
        self.ax_pattern_stability.set_ylabel('Recognition Rate (%)')
        
        # Calculate rolling recognition rate
        window_size = 20
        if len(attractor_data) >= window_size:
            recognition_rates = []
            windows = []
            
            for i in range(window_size, len(attractor_data) + 1):
                window_data = attractor_data[i-window_size:i]
                recognized_count = sum(1 for entry in window_data if entry['pattern_recognized'])
                rate = (recognized_count / window_size) * 100
                recognition_rates.append(rate)
                windows.append(i)
            
            self.ax_pattern_stability.plot(windows, recognition_rates, 'purple', linewidth=2, marker='s', markersize=3)
            self.ax_pattern_stability.fill_between(windows, recognition_rates, alpha=0.3, color='purple')
            self.ax_pattern_stability.set_ylim(0, 100)
            self.ax_pattern_stability.grid(True, alpha=0.3)
        else:
            self.ax_pattern_stability.text(0.5, 0.5, 'Insufficient data\nfor stability analysis', 
                                         ha='center', va='center', transform=self.ax_pattern_stability.transAxes)
    
    def _plot_behavior_clusters(self, behavior_data):
        """Plot behavioral clustering analysis."""
        self.ax_behavior_clusters.set_title('Behavioral Clustering')
        
        if behavior_data:
            # Extract behavioral features
            repetition_rates = [entry['intention_repetition_rate'] for entry in behavior_data]
            tool_entropies = [entry['tool_preference_entropy'] for entry in behavior_data]
            
            # Create scatter plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(repetition_rates)))
            scatter = self.ax_behavior_clusters.scatter(repetition_rates, tool_entropies, 
                                                      c=colors, alpha=0.6, s=50)
            
            self.ax_behavior_clusters.set_xlabel('Intention Repetition Rate')
            self.ax_behavior_clusters.set_ylabel('Tool Preference Entropy')
            self.ax_behavior_clusters.grid(True, alpha=0.3)
            
            # Add quadrant labels
            self.ax_behavior_clusters.axhline(y=np.mean(tool_entropies), color='gray', linestyle='--', alpha=0.5)
            self.ax_behavior_clusters.axvline(x=np.mean(repetition_rates), color='gray', linestyle='--', alpha=0.5)
        else:
            self.ax_behavior_clusters.text(0.5, 0.5, 'No behavioral data', ha='center', va='center', 
                                         transform=self.ax_behavior_clusters.transAxes)
    
    def _plot_intention_patterns(self, behavior_data):
        """Plot intention pattern analysis."""
        self.ax_intention_patterns.set_title('Dominant Intention Patterns')
        
        if behavior_data:
            # Count dominant intentions
            intention_counts = defaultdict(int)
            for entry in behavior_data:
                intention = entry.get('dominant_intention_category', 'Unknown')
                intention_counts[intention] += 1
            
            if intention_counts:
                intentions = list(intention_counts.keys())
                counts = list(intention_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(intentions)))
                
                wedges, texts, autotexts = self.ax_intention_patterns.pie(counts, labels=intentions, 
                                                                        colors=colors, autopct='%1.1f%%')
        else:
            self.ax_intention_patterns.text(0.5, 0.5, 'No intention data', ha='center', va='center', 
                                          transform=self.ax_intention_patterns.transAxes)
    
    def _plot_reward_attractors(self, attractor_data, behavior_data):
        """Plot reward-based attractor analysis."""
        self.ax_reward_attractors.set_title('Reward vs Attractor Strength')
        self.ax_reward_attractors.set_xlabel('Reward Trend')
        self.ax_reward_attractors.set_ylabel('Attractor Strength')
        
        if behavior_data and attractor_data:
            # Align data by cycle
            reward_trends = []
            attractor_strengths = []
            
            behavior_cycles = {entry['cycle']: entry for entry in behavior_data}
            
            for attractor_entry in attractor_data:
                cycle = attractor_entry['cycle']
                if cycle in behavior_cycles:
                    reward_trends.append(behavior_cycles[cycle]['reward_trend'])
                    attractor_strengths.append(attractor_entry['strongest_attractor_strength'])
            
            if reward_trends and attractor_strengths:
                colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in reward_trends]
                self.ax_reward_attractors.scatter(reward_trends, attractor_strengths, c=colors, alpha=0.6, s=40)
                self.ax_reward_attractors.grid(True, alpha=0.3)
                
                # Add trend line
                if len(reward_trends) > 1:
                    z = np.polyfit(reward_trends, attractor_strengths, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(reward_trends), max(reward_trends), 100)
                    self.ax_reward_attractors.plot(x_trend, p(x_trend), 'k--', alpha=0.5, linewidth=2)
        else:
            self.ax_reward_attractors.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                                         ha='center', va='center', transform=self.ax_reward_attractors.transAxes)
    
    def _plot_phase_space(self, attractor_data):
        """Plot attractor phase space."""
        self.ax_phase_space.set_title('Attractor Phase Space')
        self.ax_phase_space.set_xlabel('Attractor Count')
        self.ax_phase_space.set_ylabel('Strongest Strength')
        
        counts = [entry['attractor_count'] for entry in attractor_data]
        strengths = [entry['strongest_attractor_strength'] for entry in attractor_data]
        cycles = [entry['cycle'] for entry in attractor_data]
        
        # Color by time progression
        colors = plt.cm.plasma(np.linspace(0, 1, len(cycles)))
        scatter = self.ax_phase_space.scatter(counts, strengths, c=colors, alpha=0.7, s=30)
        
        # Add trajectory line
        self.ax_phase_space.plot(counts, strengths, 'k-', alpha=0.3, linewidth=1)
        self.ax_phase_space.grid(True, alpha=0.3)
        
        # Add colorbar for time
        if len(cycles) > 1:
            cbar = plt.colorbar(scatter, ax=self.ax_phase_space)
            cbar.set_label('Cycle Progression')
    
    def _plot_pattern_transitions(self, behavior_data):
        """Plot pattern transition analysis."""
        self.ax_pattern_transitions.set_title('Tool Usage Transitions')
        
        if behavior_data:
            # Track tool transitions
            tool_transitions = defaultdict(int)
            prev_tool = None
            
            for entry in behavior_data:
                current_tool = entry.get('most_used_tool', 'None')
                if prev_tool and prev_tool != current_tool:
                    transition = f"{prev_tool} → {current_tool}"
                    tool_transitions[transition] += 1
                prev_tool = current_tool
            
            if tool_transitions:
                transitions = list(tool_transitions.keys())[:8]  # Top 8 transitions
                counts = [tool_transitions[t] for t in transitions]
                
                bars = self.ax_pattern_transitions.barh(transitions, counts, color='lightblue', alpha=0.7)
                self.ax_pattern_transitions.set_xlabel('Transition Count')
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    width = bar.get_width()
                    self.ax_pattern_transitions.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                                                   f'{count}', ha='left', va='center', fontsize=8)
        else:
            self.ax_pattern_transitions.text(0.5, 0.5, 'No transition data', ha='center', va='center', 
                                           transform=self.ax_pattern_transitions.transAxes)
    
    def _plot_summary_statistics(self, attractor_data, behavior_data):
        """Plot summary statistics."""
        self.ax_summary_stats.set_title('Attractor Summary')
        self.ax_summary_stats.axis('off')
        
        # Calculate summary statistics
        total_patterns = len(attractor_data)
        avg_count = np.mean([entry['attractor_count'] for entry in attractor_data]) if attractor_data else 0
        avg_strength = np.mean([entry['strongest_attractor_strength'] for entry in attractor_data]) if attractor_data else 0
        recognition_rate = (sum(1 for entry in attractor_data if entry['pattern_recognized']) / len(attractor_data) * 100) if attractor_data else 0
        
        # Behavioral statistics
        if behavior_data:
            avg_repetition = np.mean([entry['intention_repetition_rate'] for entry in behavior_data])
            avg_entropy = np.mean([entry['tool_preference_entropy'] for entry in behavior_data])
        else:
            avg_repetition = 0
            avg_entropy = 0
        
        stats_text = []
        stats_text.append(f"Total Cycles: {total_patterns}")
        stats_text.append(f"Avg Attractors: {avg_count:.1f}")
        stats_text.append(f"Avg Strength: {avg_strength:.3f}")
        stats_text.append(f"Recognition Rate: {recognition_rate:.1f}%")
        stats_text.append(f"Intention Repetition: {avg_repetition:.3f}")
        stats_text.append(f"Tool Entropy: {avg_entropy:.3f}")
        
        # Stability assessment
        if avg_strength > 0.7:
            stability = "High Stability"
            color = 'green'
        elif avg_strength > 0.4:
            stability = "Medium Stability"
            color = 'orange'
        else:
            stability = "Low Stability"
            color = 'red'
        
        stats_text.append(f"System Stability: {stability}")
        
        y_pos = 0.9
        for i, stat in enumerate(stats_text):
            text_color = color if i == len(stats_text) - 1 else 'black'
            self.ax_summary_stats.text(0.1, y_pos, stat, transform=self.ax_summary_stats.transAxes, 
                                     fontsize=11, fontweight='bold', color=text_color)
            y_pos -= 0.12
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        for ax in [self.ax_pattern_count, self.ax_strength_evolution, self.ax_pattern_stability,
                  self.ax_behavior_clusters, self.ax_intention_patterns, self.ax_reward_attractors,
                  self.ax_phase_space, self.ax_pattern_transitions, self.ax_summary_stats]:
            ax.clear()
            ax.text(0.5, 0.5, 'No attractor pattern data available\n\nStart an agent session to see\nbehavioral pattern analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightsteelblue", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_pattern_count, self.ax_strength_evolution, self.ax_pattern_stability,
                  self.ax_behavior_clusters, self.ax_intention_patterns, self.ax_reward_attractors,
                  self.ax_phase_space, self.ax_pattern_transitions, self.ax_summary_stats]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()