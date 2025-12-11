"""
Thread Management Tab

Thread monitoring and management interface.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict
from datetime import datetime, timedelta

from .base_tab import BaseTab


class ThreadManagementTab(BaseTab):
    """Thread management and monitoring tab."""
    
    def __init__(self, notebook, thread_manager):
        """Initialize thread management tab."""
        self.thread_manager = thread_manager
        super().__init__(notebook, "Thread Management", "🧵")
        self._create_thread_display()
    
    def _create_thread_display(self):
        """Create the thread management display."""
        # Create figure with subplots
        fig = Figure(figsize=(16, 12), facecolor='white')
        fig.suptitle('Thread Management Dashboard', fontsize=16, fontweight='bold')
        
        # Create subplots in a 3x3 grid
        self.ax_thread_count = fig.add_subplot(3, 3, 1)
        self.ax_thread_types = fig.add_subplot(3, 3, 2)
        self.ax_thread_states = fig.add_subplot(3, 3, 3)
        self.ax_cpu_usage = fig.add_subplot(3, 3, 4)
        self.ax_memory_usage = fig.add_subplot(3, 3, 5)
        self.ax_thread_lifecycle = fig.add_subplot(3, 3, 6)
        self.ax_pool_status = fig.add_subplot(3, 3, 7)
        self.ax_error_tracking = fig.add_subplot(3, 3, 8)
        self.ax_summary_stats = fig.add_subplot(3, 3, 9)
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def update_display(self):
        """Update thread display with current data."""
        try:
            # Clear all axes
            for ax in [self.ax_thread_count, self.ax_thread_types, self.ax_thread_states,
                      self.ax_cpu_usage, self.ax_memory_usage, self.ax_thread_lifecycle,
                      self.ax_pool_status, self.ax_error_tracking, self.ax_summary_stats]:
                ax.clear()
            
            if not self.thread_manager:
                self._show_no_manager_message()
                return
            
            # Get thread data
            all_threads = self.thread_manager.get_all_threads()
            active_threads = self.thread_manager.get_active_threads()
            
            if not all_threads:
                self._show_no_threads_message()
                return
            
            # Plot thread analysis
            self._plot_thread_count_evolution(all_threads, active_threads)
            self._plot_thread_types_distribution(all_threads)
            self._plot_thread_states(all_threads)
            self._plot_cpu_usage_by_thread(active_threads)
            self._plot_memory_usage_by_thread(active_threads)
            self._plot_thread_lifecycle(all_threads)
            self._plot_thread_pool_status()
            self._plot_error_tracking(all_threads)
            self._plot_summary_statistics(all_threads, active_threads)
            
            # Refresh canvas
            if self.canvas:
                self.canvas.draw()
                
        except Exception as e:
            self._show_error_message(f"Error updating thread display: {str(e)}")
    
    def _plot_thread_count_evolution(self, all_threads, active_threads):
        """Plot thread count over time."""
        self.ax_thread_count.set_title('Thread Count Evolution')
        self.ax_thread_count.set_xlabel('Time')
        self.ax_thread_count.set_ylabel('Thread Count')
        
        # Simple current state display
        total_count = len(all_threads)
        active_count = len(active_threads)
        terminated_count = total_count - active_count
        
        categories = ['Total', 'Active', 'Terminated']
        counts = [total_count, active_count, terminated_count]
        colors = ['blue', 'green', 'red']
        
        bars = self.ax_thread_count.bar(categories, counts, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.ax_thread_count.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_thread_types_distribution(self, all_threads):
        """Plot distribution of thread types."""
        self.ax_thread_types.set_title('Thread Types Distribution')
        
        type_counts = {}
        for thread in all_threads:
            thread_type = thread.thread_type
            type_counts[thread_type] = type_counts.get(thread_type, 0) + 1
        
        if type_counts:
            labels = list(type_counts.keys())
            sizes = list(type_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = self.ax_thread_types.pie(sizes, labels=labels, colors=colors, 
                                                              autopct='%1.1f%%', startangle=90)
    
    def _plot_thread_states(self, all_threads):
        """Plot thread states distribution."""
        self.ax_thread_states.set_title('Thread States')
        
        state_counts = {}
        for thread in all_threads:
            state = thread.state.value if hasattr(thread.state, 'value') else str(thread.state)
            state_counts[state] = state_counts.get(state, 0) + 1
        
        if state_counts:
            states = list(state_counts.keys())
            counts = list(state_counts.values())
            
            # Color code by state
            color_map = {
                'running': 'green',
                'waiting': 'orange', 
                'blocked': 'red',
                'terminated': 'gray',
                'created': 'blue'
            }
            colors = [color_map.get(state.lower(), 'lightblue') for state in states]
            
            bars = self.ax_thread_states.bar(states, counts, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.ax_thread_states.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                         f'{count}', ha='center', va='bottom', fontsize=8)
            
            plt.setp(self.ax_thread_states.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_cpu_usage_by_thread(self, active_threads):
        """Plot CPU usage by thread."""
        self.ax_cpu_usage.set_title('CPU Usage by Thread')
        self.ax_cpu_usage.set_ylabel('CPU Time (seconds)')
        
        if active_threads:
            # Get top 10 threads by CPU usage
            sorted_threads = sorted(active_threads, key=lambda t: t.cpu_time, reverse=True)[:10]
            
            if sorted_threads:
                names = [t.name[:15] + '...' if len(t.name) > 15 else t.name for t in sorted_threads]
                cpu_times = [t.cpu_time for t in sorted_threads]
                
                bars = self.ax_cpu_usage.barh(names, cpu_times, color='lightcoral', alpha=0.7)
                
                # Add value labels
                for bar, cpu_time in zip(bars, cpu_times):
                    width = bar.get_width()
                    self.ax_cpu_usage.text(width + max(cpu_times)*0.01, bar.get_y() + bar.get_height()/2.,
                                         f'{cpu_time:.2f}s', ha='left', va='center', fontsize=8)
        else:
            self.ax_cpu_usage.text(0.5, 0.5, 'No active threads', ha='center', va='center', 
                                 transform=self.ax_cpu_usage.transAxes)
    
    def _plot_memory_usage_by_thread(self, active_threads):
        """Plot memory usage by thread."""
        self.ax_memory_usage.set_title('Memory Usage by Thread')
        self.ax_memory_usage.set_ylabel('Memory (MB)')
        
        if active_threads:
            # Get top 10 threads by memory usage
            sorted_threads = sorted(active_threads, key=lambda t: t.memory_usage, reverse=True)[:10]
            
            if sorted_threads and any(t.memory_usage > 0 for t in sorted_threads):
                names = [t.name[:15] + '...' if len(t.name) > 15 else t.name for t in sorted_threads]
                memory_mb = [t.memory_usage / (1024*1024) for t in sorted_threads]
                
                bars = self.ax_memory_usage.barh(names, memory_mb, color='lightblue', alpha=0.7)
                
                # Add value labels
                for bar, mem_mb in zip(bars, memory_mb):
                    width = bar.get_width()
                    self.ax_memory_usage.text(width + max(memory_mb)*0.01, bar.get_y() + bar.get_height()/2.,
                                            f'{mem_mb:.1f}MB', ha='left', va='center', fontsize=8)
            else:
                self.ax_memory_usage.text(0.5, 0.5, 'No memory data available', ha='center', va='center', 
                                        transform=self.ax_memory_usage.transAxes)
        else:
            self.ax_memory_usage.text(0.5, 0.5, 'No active threads', ha='center', va='center', 
                                    transform=self.ax_memory_usage.transAxes)
    
    def _plot_thread_lifecycle(self, all_threads):
        """Plot thread lifecycle information."""
        self.ax_thread_lifecycle.set_title('Thread Lifecycle')
        
        if all_threads:
            # Calculate thread lifetimes
            now = datetime.now()
            lifetimes = []
            names = []
            
            for thread in all_threads[:10]:  # Show top 10
                if thread.created_time:
                    lifetime = (now - thread.created_time).total_seconds()
                    lifetimes.append(lifetime)
                    name = thread.name[:15] + '...' if len(thread.name) > 15 else thread.name
                    names.append(name)
            
            if lifetimes:
                colors = ['green' if thread.is_alive else 'red' for thread in all_threads[:10]]
                bars = self.ax_thread_lifecycle.barh(names, lifetimes, color=colors, alpha=0.7)
                
                self.ax_thread_lifecycle.set_xlabel('Lifetime (seconds)')
                
                # Add value labels
                for bar, lifetime in zip(bars, lifetimes):
                    width = bar.get_width()
                    self.ax_thread_lifecycle.text(width + max(lifetimes)*0.01, bar.get_y() + bar.get_height()/2.,
                                                f'{lifetime:.1f}s', ha='left', va='center', fontsize=8)
        else:
            self.ax_thread_lifecycle.text(0.5, 0.5, 'No thread data', ha='center', va='center', 
                                        transform=self.ax_thread_lifecycle.transAxes)
    
    def _plot_thread_pool_status(self):
        """Plot thread pool status."""
        self.ax_pool_status.set_title('Thread Pool Status')
        
        if hasattr(self.thread_manager, 'get_all_thread_pools'):
            pools = self.thread_manager.get_all_thread_pools()
            
            if pools:
                pool_names = list(pools.keys())
                max_workers = [pool_info.get('max_workers', 0) for pool_info in pools.values()]
                
                bars = self.ax_pool_status.bar(pool_names, max_workers, color='lightgreen', alpha=0.7)
                self.ax_pool_status.set_ylabel('Max Workers')
                
                # Add value labels
                for bar, workers in zip(bars, max_workers):
                    height = bar.get_height()
                    self.ax_pool_status.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                           f'{workers}', ha='center', va='bottom', fontsize=8)
                
                plt.setp(self.ax_pool_status.get_xticklabels(), rotation=45, ha='right')
            else:
                self.ax_pool_status.text(0.5, 0.5, 'No thread pools registered', ha='center', va='center', 
                                       transform=self.ax_pool_status.transAxes)
        else:
            self.ax_pool_status.text(0.5, 0.5, 'Thread pool info not available', ha='center', va='center', 
                                   transform=self.ax_pool_status.transAxes)
    
    def _plot_error_tracking(self, all_threads):
        """Plot error tracking information."""
        self.ax_error_tracking.set_title('Thread Error Tracking')
        
        error_threads = [t for t in all_threads if t.error_count > 0]
        
        if error_threads:
            names = [t.name[:15] + '...' if len(t.name) > 15 else t.name for t in error_threads[:10]]
            error_counts = [t.error_count for t in error_threads[:10]]
            
            colors = ['red' if count > 5 else 'orange' if count > 2 else 'yellow' for count in error_counts]
            bars = self.ax_error_tracking.bar(names, error_counts, color=colors, alpha=0.7)
            
            self.ax_error_tracking.set_ylabel('Error Count')
            
            # Add value labels
            for bar, count in zip(bars, error_counts):
                height = bar.get_height()
                self.ax_error_tracking.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                          f'{count}', ha='center', va='bottom', fontsize=8)
            
            plt.setp(self.ax_error_tracking.get_xticklabels(), rotation=45, ha='right')
        else:
            self.ax_error_tracking.text(0.5, 0.5, '✅ No thread errors', ha='center', va='center', 
                                      transform=self.ax_error_tracking.transAxes, color='green', fontsize=12)
    
    def _plot_summary_statistics(self, all_threads, active_threads):
        """Plot summary statistics."""
        self.ax_summary_stats.set_title('Thread Summary')
        self.ax_summary_stats.axis('off')
        
        # Calculate statistics
        total_threads = len(all_threads)
        active_count = len(active_threads)
        total_cpu_time = sum(t.cpu_time for t in all_threads)
        total_memory = sum(t.memory_usage for t in all_threads) / (1024*1024)  # MB
        total_errors = sum(t.error_count for t in all_threads)
        
        # Thread types breakdown
        type_counts = {}
        for thread in all_threads:
            thread_type = thread.thread_type
            type_counts[thread_type] = type_counts.get(thread_type, 0) + 1
        
        stats_text = []
        stats_text.append(f"Total Threads: {total_threads}")
        stats_text.append(f"Active Threads: {active_count}")
        stats_text.append(f"Total CPU Time: {total_cpu_time:.2f}s")
        stats_text.append(f"Total Memory: {total_memory:.1f}MB")
        stats_text.append(f"Total Errors: {total_errors}")
        stats_text.append("")
        stats_text.append("Thread Types:")
        for thread_type, count in type_counts.items():
            stats_text.append(f"  {thread_type}: {count}")
        
        y_pos = 0.95
        for stat in stats_text:
            color = 'red' if 'Error' in stat and total_errors > 0 else 'black'
            font_weight = 'bold' if stat and not stat.startswith('  ') else 'normal'
            self.ax_summary_stats.text(0.05, y_pos, stat, transform=self.ax_summary_stats.transAxes, 
                                     fontsize=10, color=color, fontweight=font_weight)
            y_pos -= 0.08
    
    def _show_no_manager_message(self):
        """Show message when thread manager is not available."""
        for ax in [self.ax_thread_count, self.ax_thread_types, self.ax_thread_states,
                  self.ax_cpu_usage, self.ax_memory_usage, self.ax_thread_lifecycle,
                  self.ax_pool_status, self.ax_error_tracking, self.ax_summary_stats]:
            ax.clear()
            ax.text(0.5, 0.5, 'Thread manager not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_no_threads_message(self):
        """Show message when no threads are registered."""
        for ax in [self.ax_thread_count, self.ax_thread_types, self.ax_thread_states,
                  self.ax_cpu_usage, self.ax_memory_usage, self.ax_thread_lifecycle,
                  self.ax_pool_status, self.ax_error_tracking, self.ax_summary_stats]:
            ax.clear()
            ax.text(0.5, 0.5, 'No threads registered\n\nStart an agent session to see\nthread management and monitoring', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_thread_count, self.ax_thread_types, self.ax_thread_states,
                  self.ax_cpu_usage, self.ax_memory_usage, self.ax_thread_lifecycle,
                  self.ax_pool_status, self.ax_error_tracking, self.ax_summary_stats]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()