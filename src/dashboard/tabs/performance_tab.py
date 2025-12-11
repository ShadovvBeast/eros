"""
Performance Metrics Tab

Comprehensive performance analysis and trends.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import deque

from .base_tab import BaseTab


class PerformanceTab(BaseTab):
    """Performance metrics tab."""
    
    def __init__(self, notebook, collector):
        """Initialize performance tab."""
        self.collector = collector
        super().__init__(notebook, "Performance", "⚡")
        self._create_performance_display()
    
    def _create_performance_display(self):
        """Create the performance analysis display."""
        # Create figure with subplots
        fig = Figure(figsize=(15, 10), facecolor='white')
        fig.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.ax_cycle_times = fig.add_subplot(2, 3, 1)
        self.ax_tool_success = fig.add_subplot(2, 3, 2)
        self.ax_memory_events = fig.add_subplot(2, 3, 3)
        self.ax_phase_breakdown = fig.add_subplot(2, 3, 4)
        self.ax_efficiency_trend = fig.add_subplot(2, 3, 5)
        self.ax_summary_stats = fig.add_subplot(2, 3, 6)
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def update_display(self):
        """Update performance display with current data."""
        try:
            # Clear all axes
            for ax in [self.ax_cycle_times, self.ax_tool_success, self.ax_memory_events, 
                      self.ax_phase_breakdown, self.ax_efficiency_trend, self.ax_summary_stats]:
                ax.clear()
            
            # Check if we have performance data
            if not hasattr(self.collector, 'metrics'):
                self._show_no_data_message()
                return
            
            metrics = self.collector.metrics
            
            # Debug: Check what metrics we have
            available_metrics = list(metrics.keys())
            total_data_points = sum(len(metrics[key]) if isinstance(metrics[key], list) else 0 for key in metrics)
            
            # If we have some data, show it
            if total_data_points > 0:
                print(f"📊 Performance tab found {total_data_points} data points in metrics: {available_metrics}")
            else:
                print(f"📊 Performance tab: No data in metrics: {available_metrics}")
                self._show_no_data_message()
                return
            
            # Plot 1: Cycle Times Over Time
            self._plot_cycle_times(metrics)
            
            # Plot 2: Tool Success Rates
            self._plot_tool_success(metrics)
            
            # Plot 3: Memory Events
            self._plot_memory_events(metrics)
            
            # Plot 4: Phase Breakdown
            self._plot_phase_breakdown(metrics)
            
            # Plot 5: Efficiency Trend
            self._plot_efficiency_trend(metrics)
            
            # Plot 6: Summary Statistics
            self._plot_summary_stats(metrics)
            
            # Refresh canvas
            if self.canvas:
                self.canvas.draw()
                
        except Exception as e:
            self._show_error_message(f"Error updating performance display: {str(e)}")
    
    def _plot_cycle_times(self, metrics):
        """Plot cycle execution times."""
        self.ax_cycle_times.set_title('Cycle Execution Times')
        self.ax_cycle_times.set_xlabel('Cycle Number')
        self.ax_cycle_times.set_ylabel('Duration (seconds)')
        
        if 'cycle_timings' in metrics and metrics['cycle_timings']:
            cycles = [timing['cycle'] for timing in metrics['cycle_timings']]
            durations = [timing['total_duration'] for timing in metrics['cycle_timings']]
            
            self.ax_cycle_times.plot(cycles, durations, 'b-', linewidth=2, alpha=0.7)
            self.ax_cycle_times.scatter(cycles, durations, c='blue', alpha=0.5, s=20)
            
            # Add average line
            if durations:
                avg_duration = np.mean(durations)
                self.ax_cycle_times.axhline(y=avg_duration, color='red', linestyle='--', 
                                          label=f'Average: {avg_duration:.3f}s')
                self.ax_cycle_times.legend()
        else:
            self.ax_cycle_times.text(0.5, 0.5, 'No cycle timing data', ha='center', va='center', 
                                   transform=self.ax_cycle_times.transAxes)
        
        self.ax_cycle_times.grid(True, alpha=0.3)
    
    def _plot_tool_success(self, metrics):
        """Plot tool success rates."""
        self.ax_tool_success.set_title('Tool Success Rates')
        
        if 'tool_usage' in metrics and metrics['tool_usage']:
            tool_stats = {}
            for usage in metrics['tool_usage']:
                tool_name = usage['tool_name']
                success = usage['success']
                
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {'total': 0, 'success': 0}
                
                tool_stats[tool_name]['total'] += 1
                if success:
                    tool_stats[tool_name]['success'] += 1
            
            if tool_stats:
                tools = list(tool_stats.keys())
                success_rates = [tool_stats[tool]['success'] / tool_stats[tool]['total'] * 100 
                               for tool in tools]
                
                colors = ['green' if rate > 80 else 'orange' if rate > 50 else 'red' 
                         for rate in success_rates]
                
                bars = self.ax_tool_success.bar(tools, success_rates, color=colors, alpha=0.7)
                self.ax_tool_success.set_ylabel('Success Rate (%)')
                self.ax_tool_success.set_ylim(0, 100)
                plt.setp(self.ax_tool_success.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, rate in zip(bars, success_rates):
                    height = bar.get_height()
                    self.ax_tool_success.text(bar.get_x() + bar.get_width()/2., height + 1,
                                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            self.ax_tool_success.text(0.5, 0.5, 'No tool usage data', ha='center', va='center', 
                                    transform=self.ax_tool_success.transAxes)
    
    def _plot_memory_events(self, metrics):
        """Plot memory storage events over time."""
        self.ax_memory_events.set_title('Memory Storage Events')
        self.ax_memory_events.set_xlabel('Event Number')
        self.ax_memory_events.set_ylabel('Salience Score')
        
        if 'memory_events' in metrics and metrics['memory_events']:
            saliences = [event['salience'] for event in metrics['memory_events']]
            event_numbers = list(range(len(saliences)))
            
            self.ax_memory_events.scatter(event_numbers, saliences, c='purple', alpha=0.6, s=30)
            self.ax_memory_events.plot(event_numbers, saliences, 'purple', alpha=0.3, linewidth=1)
            
            # Add threshold line (typical salience threshold)
            threshold = 0.5
            self.ax_memory_events.axhline(y=threshold, color='red', linestyle='--', 
                                        label=f'Threshold: {threshold}')
            self.ax_memory_events.legend()
        else:
            self.ax_memory_events.text(0.5, 0.5, 'No memory events data', ha='center', va='center', 
                                     transform=self.ax_memory_events.transAxes)
        
        self.ax_memory_events.grid(True, alpha=0.3)
    
    def _plot_phase_breakdown(self, metrics):
        """Plot average time breakdown by phase."""
        self.ax_phase_breakdown.set_title('Average Phase Duration Breakdown')
        
        if 'cycle_timings' in metrics and metrics['cycle_timings']:
            phase_totals = {}
            phase_counts = {}
            
            for timing in metrics['cycle_timings']:
                if 'phase_timings' in timing:
                    for phase, duration in timing['phase_timings'].items():
                        if phase not in phase_totals:
                            phase_totals[phase] = 0
                            phase_counts[phase] = 0
                        phase_totals[phase] += duration
                        phase_counts[phase] += 1
            
            if phase_totals:
                phases = list(phase_totals.keys())
                avg_durations = [phase_totals[phase] / phase_counts[phase] for phase in phases]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
                wedges, texts, autotexts = self.ax_phase_breakdown.pie(avg_durations, labels=phases, 
                                                                     colors=colors, autopct='%1.1f%%')
        else:
            self.ax_phase_breakdown.text(0.5, 0.5, 'No phase timing data', ha='center', va='center', 
                                       transform=self.ax_phase_breakdown.transAxes)
    
    def _plot_efficiency_trend(self, metrics):
        """Plot efficiency trend over time."""
        self.ax_efficiency_trend.set_title('Efficiency Trend (Cycles per Minute)')
        self.ax_efficiency_trend.set_xlabel('Time Window')
        self.ax_efficiency_trend.set_ylabel('Cycles per Minute')
        
        if 'cycle_timings' in metrics and metrics['cycle_timings']:
            # Calculate rolling efficiency (cycles per minute in windows of 10)
            window_size = 10
            timings = metrics['cycle_timings']
            
            if len(timings) >= window_size:
                efficiencies = []
                windows = []
                
                for i in range(window_size, len(timings) + 1):
                    window_timings = timings[i-window_size:i]
                    total_time = sum(t['total_duration'] for t in window_timings)
                    cycles_per_minute = (window_size / total_time) * 60 if total_time > 0 else 0
                    efficiencies.append(cycles_per_minute)
                    windows.append(i)
                
                self.ax_efficiency_trend.plot(windows, efficiencies, 'g-', linewidth=2, marker='o')
                self.ax_efficiency_trend.grid(True, alpha=0.3)
        else:
            self.ax_efficiency_trend.text(0.5, 0.5, 'Insufficient data for trend', ha='center', va='center', 
                                        transform=self.ax_efficiency_trend.transAxes)
    
    def _plot_summary_stats(self, metrics):
        """Plot summary statistics."""
        self.ax_summary_stats.set_title('Performance Summary')
        self.ax_summary_stats.axis('off')
        
        # Calculate summary statistics
        summary = self.collector.get_metrics_summary()
        
        stats_text = []
        stats_text.append(f"Runtime: {summary.get('runtime_seconds', 0):.1f}s")
        stats_text.append(f"Total Cycles: {summary.get('total_cycles', 0)}")
        stats_text.append(f"Avg Cycle Time: {summary.get('avg_cycle_time', 0):.3f}s")
        stats_text.append(f"Memory Events: {summary.get('total_memory_events', 0)}")
        stats_text.append(f"Tool Calls: {summary.get('total_tool_calls', 0)}")
        stats_text.append(f"Errors: {summary.get('total_errors', 0)}")
        
        # Display as formatted text
        y_pos = 0.9
        for stat in stats_text:
            self.ax_summary_stats.text(0.1, y_pos, stat, transform=self.ax_summary_stats.transAxes, 
                                     fontsize=12, fontweight='bold')
            y_pos -= 0.15
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        for ax in [self.ax_cycle_times, self.ax_tool_success, self.ax_memory_events, 
                  self.ax_phase_breakdown, self.ax_efficiency_trend, self.ax_summary_stats]:
            ax.clear()
            ax.text(0.5, 0.5, 'No performance data available\n\nStart an agent session to see\ncomprehensive performance metrics', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_cycle_times, self.ax_tool_success, self.ax_memory_events, 
                  self.ax_phase_breakdown, self.ax_efficiency_trend, self.ax_summary_stats]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()