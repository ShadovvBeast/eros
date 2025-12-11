"""
System Health Tab

Enhanced system health overview with hardware integration.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Optional

try:
    from ...core.logging_config import InstrumentationCollector
    from ...monitoring.hardware_monitor import HardwareMonitor
except ImportError:
    from core.logging_config import InstrumentationCollector
    from monitoring.hardware_monitor import HardwareMonitor

from .base_tab import BaseTab


class HealthTab(BaseTab):
    """System health overview tab."""
    
    def __init__(self, notebook, collector: InstrumentationCollector, 
                 hardware_monitor: HardwareMonitor):
        """Initialize health tab."""
        self.collector = collector
        self.hardware_monitor = hardware_monitor
        
        super().__init__(notebook, "System Health", "🏥")
        self._create_health_display()
    
    def _create_health_display(self):
        """Create enhanced system health overview."""
        # Create figure with subplots
        fig = Figure(figsize=(14, 10), facecolor='white')
        
        # System overview (top left)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.set_title("System Status", fontweight='bold')
        
        # Hardware metrics (top middle)
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.set_title("Hardware Overview", fontweight='bold')
        
        # Performance trends (top right)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_title("Performance Trends", fontweight='bold')
        
        # Agent status (bottom left)
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.set_title("Agent Status", fontweight='bold')
        
        # Memory usage (bottom middle)
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.set_title("Memory Usage", fontweight='bold')
        
        # System alerts (bottom right)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.set_title("System Alerts", fontweight='bold')
        
        # Store axes for updates
        self.axes = {
            'system': ax1,
            'hardware': ax2,
            'performance': ax3,
            'agent': ax4,
            'memory': ax5,
            'alerts': ax6
        }
        
        fig.tight_layout(pad=3.0)
        self._add_matplotlib_canvas(fig)
        
        # Initial display
        self.update_display()
    
    def update_display(self):
        """Update health display with current data."""
        try:
            # Clear all axes
            for ax in self.axes.values():
                ax.clear()
            
            self._update_system_status()
            self._update_hardware_overview()
            self._update_performance_trends()
            self._update_agent_status()
            self._update_memory_usage()
            self._update_system_alerts()
            
            # Refresh canvas
            self.refresh()
            
        except Exception as e:
            print(f"Health tab update error: {e}")
    
    def _update_system_status(self):
        """Update system status display."""
        ax = self.axes['system']
        ax.set_title("System Status", fontweight='bold')
        
        # System components status
        components = ['Agent', 'Pathos', 'Memory', 'Logos', 'Tools', 'Hardware']
        statuses = [1, 1, 1, 1, 1, 1 if self.hardware_monitor.is_monitoring else 0]
        colors = ['green' if s else 'red' for s in statuses]
        
        bars = ax.barh(components, statuses, color=colors, alpha=0.7)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel('Status')
        
        # Add status text
        for i, (bar, status) in enumerate(zip(bars, statuses)):
            text = "Online" if status else "Offline"
            ax.text(0.6, i, text, ha='center', va='center', fontweight='bold', color='white')
    
    def _update_hardware_overview(self):
        """Update hardware overview."""
        ax = self.axes['hardware']
        ax.set_title("Hardware Overview", fontweight='bold')
        
        if self.hardware_monitor.is_monitoring:
            metrics = self.hardware_monitor.get_current_metrics()
            
            # CPU and Memory usage
            cpu_usage = metrics.get('cpu_percent', 0)
            memory_usage = metrics.get('memory_percent', 0)
            
            # Create gauge-style display
            categories = ['CPU', 'Memory']
            values = [cpu_usage, memory_usage]
            colors = ['red' if v > 80 else 'orange' if v > 60 else 'green' for v in values]
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Usage (%)')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Hardware monitoring offline', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _update_performance_trends(self):
        """Update performance trends."""
        ax = self.axes['performance']
        ax.set_title("Performance Trends", fontweight='bold')
        
        # Get recent performance data
        if hasattr(self.collector, 'get_recent_metrics'):
            metrics = self.collector.get_recent_metrics(50)
            if metrics:
                times = list(range(len(metrics)))
                cycle_times = [m.get('cycle_time', 0) for m in metrics]
                
                ax.plot(times, cycle_times, 'b-', linewidth=2, label='Cycle Time')
                ax.set_xlabel('Recent Cycles')
                ax.set_ylabel('Time (s)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No performance data available', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Performance monitoring unavailable', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _update_agent_status(self):
        """Update agent status."""
        ax = self.axes['agent']
        ax.set_title("Agent Status", fontweight='bold')
        
        # Agent layer status
        layers = ['Logos', 'Pathos', 'Memory', 'Ethos', 'Tools']
        # Mock status for now - in real implementation, get from agent
        statuses = [1, 1, 1, 1, 1]
        
        # Pie chart of layer health
        colors = ['lightgreen' if s else 'lightcoral' for s in statuses]
        ax.pie([1]*len(layers), labels=layers, colors=colors, autopct='',
               startangle=90, counterclock=False)
        ax.axis('equal')
    
    def _update_memory_usage(self):
        """Update memory usage display."""
        ax = self.axes['memory']
        ax.set_title("Memory Usage", fontweight='bold')
        
        if self.hardware_monitor.is_monitoring:
            metrics = self.hardware_monitor.get_current_metrics()
            
            # Memory breakdown
            total_memory = metrics.get('memory_total', 0) / (1024**3)  # GB
            used_memory = metrics.get('memory_used', 0) / (1024**3)   # GB
            free_memory = total_memory - used_memory
            
            # Pie chart
            sizes = [used_memory, free_memory]
            labels = ['Used', 'Free']
            colors = ['lightcoral', 'lightgreen']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            
            # Add total memory info
            ax.text(0, -1.3, f'Total: {total_memory:.1f} GB', 
                   ha='center', transform=ax.transData)
        else:
            ax.text(0.5, 0.5, 'Memory data unavailable', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _update_system_alerts(self):
        """Update system alerts."""
        ax = self.axes['alerts']
        ax.set_title("System Alerts", fontweight='bold')
        
        # Mock alerts - in real implementation, get from monitoring systems
        alerts = []
        
        if self.hardware_monitor.is_monitoring:
            metrics = self.hardware_monitor.get_current_metrics()
            cpu_usage = metrics.get('cpu_percent', 0)
            memory_usage = metrics.get('memory_percent', 0)
            
            if cpu_usage > 80:
                alerts.append(f"High CPU usage: {cpu_usage:.1f}%")
            if memory_usage > 80:
                alerts.append(f"High memory usage: {memory_usage:.1f}%")
        
        if not alerts:
            alerts = ["All systems normal"]
        
        # Display alerts as text
        ax.axis('off')
        for i, alert in enumerate(alerts[:5]):  # Show max 5 alerts
            color = 'red' if 'High' in alert else 'green'
            ax.text(0.05, 0.9 - i*0.15, f"• {alert}", 
                   transform=ax.transAxes, fontsize=10, color=color,
                   verticalalignment='top')