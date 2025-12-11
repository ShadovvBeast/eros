"""
Memory Network Tab

Memory trace visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .base_tab import BaseTab


class MemoryTab(BaseTab):
    """Memory network visualization tab."""
    
    def __init__(self, notebook, memory_traces):
        """Initialize memory tab."""
        self.memory_traces = memory_traces
        super().__init__(notebook, "Memory Network", "🧠")
        self._create_memory_display()
    
    def _create_memory_display(self):
        """Create the memory visualization display."""
        # Create figure with subplots
        fig = Figure(figsize=(15, 10), facecolor='white')
        fig.suptitle('Memory Network Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.ax_trace_count = fig.add_subplot(2, 3, 1)
        self.ax_salience_dist = fig.add_subplot(2, 3, 2)
        self.ax_reward_dist = fig.add_subplot(2, 3, 3)
        self.ax_temporal = fig.add_subplot(2, 3, 4)
        self.ax_categories = fig.add_subplot(2, 3, 5)
        self.ax_summary = fig.add_subplot(2, 3, 6)
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def update_display(self):
        """Update memory display with current data."""
        try:
            # Clear all axes
            for ax in [self.ax_trace_count, self.ax_salience_dist, self.ax_reward_dist, 
                      self.ax_temporal, self.ax_categories, self.ax_summary]:
                ax.clear()
            
            # Check if we have memory traces
            if not self.memory_traces or len(self.memory_traces) == 0:
                self._show_no_data_message()
                return
            
            # Plot memory analysis
            self._plot_trace_count_over_time()
            self._plot_salience_distribution()
            self._plot_reward_distribution()
            self._plot_temporal_patterns()
            self._plot_category_breakdown()
            self._plot_summary_stats()
            
            # Refresh canvas
            if self.canvas:
                self.canvas.draw()
                
        except Exception as e:
            self._show_error_message(f"Error updating memory display: {str(e)}")
    
    def _plot_trace_count_over_time(self):
        """Plot memory trace count over time."""
        self.ax_trace_count.set_title('Memory Traces Over Time')
        self.ax_trace_count.set_xlabel('Trace Index')
        self.ax_trace_count.set_ylabel('Cumulative Count')
        
        if self.memory_traces:
            trace_indices = list(range(len(self.memory_traces)))
            cumulative_counts = list(range(1, len(self.memory_traces) + 1))
            
            self.ax_trace_count.plot(trace_indices, cumulative_counts, 'b-', linewidth=2)
            self.ax_trace_count.fill_between(trace_indices, cumulative_counts, alpha=0.3)
            self.ax_trace_count.grid(True, alpha=0.3)
    
    def _plot_salience_distribution(self):
        """Plot distribution of salience scores."""
        self.ax_salience_dist.set_title('Salience Score Distribution')
        self.ax_salience_dist.set_xlabel('Salience Score')
        self.ax_salience_dist.set_ylabel('Frequency')
        
        if self.memory_traces:
            saliences = [trace.salience for trace in self.memory_traces if hasattr(trace, 'salience')]
            
            if saliences:
                self.ax_salience_dist.hist(saliences, bins=20, alpha=0.7, color='purple', edgecolor='black')
                
                # Add mean line
                mean_salience = np.mean(saliences)
                self.ax_salience_dist.axvline(mean_salience, color='red', linestyle='--', 
                                            label=f'Mean: {mean_salience:.3f}')
                self.ax_salience_dist.legend()
    
    def _plot_reward_distribution(self):
        """Plot distribution of reward values."""
        self.ax_reward_dist.set_title('Reward Distribution')
        self.ax_reward_dist.set_xlabel('Reward Value')
        self.ax_reward_dist.set_ylabel('Frequency')
        
        if self.memory_traces:
            rewards = [trace.reward for trace in self.memory_traces if hasattr(trace, 'reward')]
            
            if rewards:
                self.ax_reward_dist.hist(rewards, bins=20, alpha=0.7, color='green', edgecolor='black')
                
                # Add mean line
                mean_reward = np.mean(rewards)
                self.ax_reward_dist.axvline(mean_reward, color='red', linestyle='--', 
                                          label=f'Mean: {mean_reward:.3f}')
                self.ax_reward_dist.legend()
    
    def _plot_temporal_patterns(self):
        """Plot temporal patterns in memory storage."""
        self.ax_temporal.set_title('Memory Storage Temporal Pattern')
        self.ax_temporal.set_xlabel('Trace Index')
        self.ax_temporal.set_ylabel('Time Between Traces')
        
        if len(self.memory_traces) > 1:
            timestamps = [trace.timestamp for trace in self.memory_traces if hasattr(trace, 'timestamp')]
            
            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps)
                trace_indices = list(range(1, len(timestamps)))
                
                self.ax_temporal.plot(trace_indices, time_diffs, 'o-', color='orange', alpha=0.7)
                self.ax_temporal.grid(True, alpha=0.3)
    
    def _plot_category_breakdown(self):
        """Plot breakdown by semantic categories."""
        self.ax_categories.set_title('Memory by Semantic Category')
        
        if self.memory_traces:
            categories = {}
            for trace in self.memory_traces:
                if hasattr(trace, 'metadata') and trace.metadata and 'intention' in trace.metadata:
                    # Extract category from intention (simplified)
                    intention = trace.metadata['intention']
                    # Simple categorization based on keywords
                    if 'explore' in intention.lower():
                        category = 'Exploration'
                    elif 'analyze' in intention.lower():
                        category = 'Analysis'
                    elif 'learn' in intention.lower():
                        category = 'Learning'
                    elif 'reflect' in intention.lower():
                        category = 'Reflection'
                    else:
                        category = 'Other'
                    
                    categories[category] = categories.get(category, 0) + 1
            
            if categories:
                labels = list(categories.keys())
                sizes = list(categories.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                wedges, texts, autotexts = self.ax_categories.pie(sizes, labels=labels, colors=colors, 
                                                                autopct='%1.1f%%', startangle=90)
    
    def _plot_summary_stats(self):
        """Plot summary statistics."""
        self.ax_summary.set_title('Memory Summary')
        self.ax_summary.axis('off')
        
        if self.memory_traces:
            total_traces = len(self.memory_traces)
            
            # Calculate statistics
            saliences = [trace.salience for trace in self.memory_traces if hasattr(trace, 'salience')]
            rewards = [trace.reward for trace in self.memory_traces if hasattr(trace, 'reward')]
            
            avg_salience = np.mean(saliences) if saliences else 0
            avg_reward = np.mean(rewards) if rewards else 0
            high_salience_count = len([s for s in saliences if s > 0.7]) if saliences else 0
            
            stats_text = []
            stats_text.append(f"Total Traces: {total_traces}")
            stats_text.append(f"Avg Salience: {avg_salience:.3f}")
            stats_text.append(f"Avg Reward: {avg_reward:.3f}")
            stats_text.append(f"High Salience: {high_salience_count}")
            stats_text.append(f"Storage Rate: {(total_traces/max(1, total_traces))*100:.1f}%")
            
            y_pos = 0.9
            for stat in stats_text:
                self.ax_summary.text(0.1, y_pos, stat, transform=self.ax_summary.transAxes, 
                                   fontsize=12, fontweight='bold')
                y_pos -= 0.15
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        for ax in [self.ax_trace_count, self.ax_salience_dist, self.ax_reward_dist, 
                  self.ax_temporal, self.ax_categories, self.ax_summary]:
            ax.clear()
            ax.text(0.5, 0.5, 'No memory traces available\n\nStart an agent session to see\nmemory formation and trace analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_trace_count, self.ax_salience_dist, self.ax_reward_dist, 
                  self.ax_temporal, self.ax_categories, self.ax_summary]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()