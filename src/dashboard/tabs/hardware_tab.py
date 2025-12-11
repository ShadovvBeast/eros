"""
Hardware Monitoring Tab

Comprehensive hardware monitoring and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import deque

from .base_tab import BaseTab


class HardwareTab(BaseTab):
    """Hardware monitoring tab."""
    
    def __init__(self, notebook, hardware_monitor):
        """Initialize hardware tab."""
        self.hardware_monitor = hardware_monitor
        super().__init__(notebook, "Hardware Monitor", "🖥️")
        self._create_hardware_display()
    
    def _create_hardware_display(self):
        """Create the hardware monitoring display."""
        # Create figure with subplots
        fig = Figure(figsize=(16, 12), facecolor='white')
        fig.suptitle('Hardware Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # Create subplots in a 3x3 grid
        self.ax_cpu = fig.add_subplot(3, 3, 1)
        self.ax_memory = fig.add_subplot(3, 3, 2)
        self.ax_disk = fig.add_subplot(3, 3, 3)
        self.ax_network = fig.add_subplot(3, 3, 4)
        self.ax_gpu = fig.add_subplot(3, 3, 5)
        self.ax_temp = fig.add_subplot(3, 3, 6)
        self.ax_process = fig.add_subplot(3, 3, 7)
        self.ax_system_info = fig.add_subplot(3, 3, 8)
        self.ax_alerts = fig.add_subplot(3, 3, 9)
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def update_display(self):
        """Update hardware display with current data."""
        try:
            # Clear all axes
            for ax in [self.ax_cpu, self.ax_memory, self.ax_disk, self.ax_network, 
                      self.ax_gpu, self.ax_temp, self.ax_process, self.ax_system_info, self.ax_alerts]:
                ax.clear()
            
            if not self.hardware_monitor.is_monitoring:
                self._show_not_monitoring_message()
                return
            
            # Get current hardware data
            summary = self.hardware_monitor.get_current_summary()
            if 'error' in summary:
                self._show_error_message(f"Hardware monitoring error: {summary['error']}")
                return
            
            metrics = summary.get('current_metrics', {})
            
            # Plot each hardware component
            self._plot_cpu_usage(metrics.get('cpu', {}))
            self._plot_memory_usage(metrics.get('memory', {}))
            self._plot_disk_usage(metrics.get('disk', {}))
            self._plot_network_usage(metrics.get('network', {}))
            self._plot_gpu_usage(metrics.get('gpu', {}))
            self._plot_temperature(metrics.get('temperature', {}))
            self._plot_process_info(metrics.get('process', {}))
            self._plot_system_info(summary.get('system_info', {}))
            self._plot_alerts(metrics)
            
            # Refresh canvas
            if self.canvas:
                self.canvas.draw()
                
        except Exception as e:
            self._show_error_message(f"Error updating hardware display: {str(e)}")
    
    def _plot_cpu_usage(self, cpu_data):
        """Plot CPU usage information."""
        self.ax_cpu.set_title('CPU Usage', fontweight='bold')
        
        if cpu_data and 'cpu_percent' in cpu_data:
            cpu_percent = cpu_data['cpu_percent']
            
            # Create a gauge-style display
            colors = ['red' if cpu_percent > 80 else 'orange' if cpu_percent > 60 else 'green']
            bars = self.ax_cpu.bar(['CPU'], [cpu_percent], color=colors, alpha=0.7)
            self.ax_cpu.set_ylim(0, 100)
            self.ax_cpu.set_ylabel('Usage (%)')
            
            # Add value label
            self.ax_cpu.text(0, cpu_percent + 2, f'{cpu_percent:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
            
            # Add frequency info if available
            if 'cpu_freq_current' in cpu_data:
                freq = cpu_data['cpu_freq_current']
                self.ax_cpu.text(0.5, 0.1, f'Freq: {freq:.0f} MHz', 
                               transform=self.ax_cpu.transAxes, ha='center', fontsize=8)
        else:
            self.ax_cpu.text(0.5, 0.5, 'No CPU data', ha='center', va='center', 
                           transform=self.ax_cpu.transAxes)
    
    def _plot_memory_usage(self, memory_data):
        """Plot memory usage information."""
        self.ax_memory.set_title('Memory Usage', fontweight='bold')
        
        if memory_data and 'virtual_total' in memory_data:
            total_gb = memory_data['virtual_total'] / (1024**3)
            used_gb = memory_data['virtual_used'] / (1024**3)
            free_gb = total_gb - used_gb
            percent = memory_data.get('virtual_percent', 0)
            
            # Pie chart
            sizes = [used_gb, free_gb]
            labels = ['Used', 'Free']
            colors = ['lightcoral' if percent > 80 else 'lightsalmon' if percent > 60 else 'lightblue', 'lightgreen']
            
            wedges, texts, autotexts = self.ax_memory.pie(sizes, labels=labels, colors=colors, 
                                                        autopct='%1.1f%%', startangle=90)
            
            # Add total memory info
            self.ax_memory.text(0.5, -1.3, f'Total: {total_gb:.1f} GB', 
                              transform=self.ax_memory.transAxes, ha='center', fontsize=8)
        else:
            self.ax_memory.text(0.5, 0.5, 'No memory data', ha='center', va='center', 
                              transform=self.ax_memory.transAxes)
    
    def _plot_disk_usage(self, disk_data):
        """Plot disk usage information."""
        self.ax_disk.set_title('Disk I/O', fontweight='bold')
        
        if disk_data and 'read_rate' in disk_data:
            read_rate = disk_data['read_rate'] / 1024  # Convert to KB/s
            write_rate = disk_data['write_rate'] / 1024  # Convert to KB/s
            
            rates = [read_rate, write_rate]
            labels = ['Read', 'Write']
            colors = ['lightblue', 'lightcoral']
            
            bars = self.ax_disk.bar(labels, rates, color=colors, alpha=0.7)
            self.ax_disk.set_ylabel('Rate (KB/s)')
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                self.ax_disk.text(bar.get_x() + bar.get_width()/2., height + max(rates)*0.01,
                                f'{rate:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            self.ax_disk.text(0.5, 0.5, 'No disk data', ha='center', va='center', 
                            transform=self.ax_disk.transAxes)
    
    def _plot_network_usage(self, network_data):
        """Plot network usage information."""
        self.ax_network.set_title('Network I/O', fontweight='bold')
        
        if network_data and 'bytes_sent_rate' in network_data:
            sent_rate = network_data['bytes_sent_rate'] / 1024  # Convert to KB/s
            recv_rate = network_data['bytes_recv_rate'] / 1024  # Convert to KB/s
            
            rates = [sent_rate, recv_rate]
            labels = ['Sent', 'Received']
            colors = ['lightgreen', 'lightblue']
            
            bars = self.ax_network.bar(labels, rates, color=colors, alpha=0.7)
            self.ax_network.set_ylabel('Rate (KB/s)')
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                self.ax_network.text(bar.get_x() + bar.get_width()/2., height + max(rates)*0.01,
                                   f'{rate:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            self.ax_network.text(0.5, 0.5, 'No network data', ha='center', va='center', 
                               transform=self.ax_network.transAxes)
    
    def _plot_gpu_usage(self, gpu_data):
        """Plot GPU usage information."""
        self.ax_gpu.set_title('GPU Status', fontweight='bold')
        
        if gpu_data and 'gpus' in gpu_data and gpu_data['gpus']:
            # If GPU data is available, show first GPU
            gpu = gpu_data['gpus'][0]
            gpu_percent = gpu.get('utilization', 0)
            
            colors = ['red' if gpu_percent > 80 else 'orange' if gpu_percent > 60 else 'green']
            bars = self.ax_gpu.bar(['GPU'], [gpu_percent], color=colors, alpha=0.7)
            self.ax_gpu.set_ylim(0, 100)
            self.ax_gpu.set_ylabel('Usage (%)')
            
            # Add value label
            self.ax_gpu.text(0, gpu_percent + 2, f'{gpu_percent:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
        else:
            self.ax_gpu.text(0.5, 0.5, 'No GPU detected\nor not available', ha='center', va='center', 
                           transform=self.ax_gpu.transAxes, fontsize=10)
    
    def _plot_temperature(self, temp_data):
        """Plot temperature information."""
        self.ax_temp.set_title('System Temperature', fontweight='bold')
        
        if temp_data and 'temperatures' in temp_data and temp_data['temperatures']:
            temps = temp_data['temperatures']
            sensors = list(temps.keys())[:5]  # Show up to 5 sensors
            values = [temps[sensor] for sensor in sensors]
            
            colors = ['red' if temp > 80 else 'orange' if temp > 60 else 'green' for temp in values]
            bars = self.ax_temp.bar(sensors, values, color=colors, alpha=0.7)
            self.ax_temp.set_ylabel('Temperature (°C)')
            plt.setp(self.ax_temp.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, temp in zip(bars, values):
                height = bar.get_height()
                self.ax_temp.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{temp:.1f}°C', ha='center', va='bottom', fontsize=8)
        else:
            self.ax_temp.text(0.5, 0.5, 'No temperature\nsensors available', ha='center', va='center', 
                            transform=self.ax_temp.transAxes, fontsize=10)
    
    def _plot_process_info(self, process_data):
        """Plot current process information."""
        self.ax_process.set_title('Current Process', fontweight='bold')
        
        if process_data and 'cpu_percent' in process_data:
            cpu_percent = process_data['cpu_percent']
            memory_percent = process_data['memory_percent']
            
            metrics = ['CPU', 'Memory']
            values = [cpu_percent, memory_percent]
            colors = ['lightblue', 'lightcoral']
            
            bars = self.ax_process.bar(metrics, values, color=colors, alpha=0.7)
            self.ax_process.set_ylabel('Usage (%)')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.ax_process.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                                   f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            self.ax_process.text(0.5, 0.5, 'No process data', ha='center', va='center', 
                               transform=self.ax_process.transAxes)
    
    def _plot_system_info(self, system_info):
        """Plot system information."""
        self.ax_system_info.set_title('System Info', fontweight='bold')
        self.ax_system_info.axis('off')
        
        if system_info and not isinstance(system_info, dict) or 'error' not in system_info:
            info_text = []
            if isinstance(system_info, dict):
                info_text.append(f"OS: {system_info.get('system', 'Unknown')}")
                info_text.append(f"CPU Cores: {system_info.get('cpu_count', 'Unknown')}")
                info_text.append(f"Architecture: {system_info.get('machine', 'Unknown')}")
                info_text.append(f"Python: {system_info.get('python_version', 'Unknown')}")
            
            y_pos = 0.9
            for info in info_text:
                self.ax_system_info.text(0.1, y_pos, info, transform=self.ax_system_info.transAxes, 
                                       fontsize=10, fontweight='bold')
                y_pos -= 0.2
        else:
            self.ax_system_info.text(0.5, 0.5, 'System info\nnot available', ha='center', va='center', 
                                   transform=self.ax_system_info.transAxes, fontsize=10)
    
    def _plot_alerts(self, metrics):
        """Plot system alerts."""
        self.ax_alerts.set_title('System Alerts', fontweight='bold')
        self.ax_alerts.axis('off')
        
        alerts = []
        
        # Check CPU
        cpu_data = metrics.get('cpu', {})
        if cpu_data and cpu_data.get('cpu_percent', 0) > 80:
            alerts.append(f"⚠️ High CPU: {cpu_data['cpu_percent']:.1f}%")
        
        # Check Memory
        memory_data = metrics.get('memory', {})
        if memory_data and memory_data.get('virtual_percent', 0) > 80:
            alerts.append(f"⚠️ High Memory: {memory_data['virtual_percent']:.1f}%")
        
        # Check Process
        process_data = metrics.get('process', {})
        if process_data and process_data.get('cpu_percent', 0) > 100:
            alerts.append(f"⚠️ High Process CPU: {process_data['cpu_percent']:.1f}%")
        
        if not alerts:
            alerts.append("✅ All systems normal")
        
        y_pos = 0.9
        for alert in alerts[:5]:  # Show up to 5 alerts
            color = 'red' if '⚠️' in alert else 'green'
            self.ax_alerts.text(0.1, y_pos, alert, transform=self.ax_alerts.transAxes, 
                              fontsize=10, color=color, fontweight='bold')
            y_pos -= 0.15
    
    def _show_not_monitoring_message(self):
        """Show message when hardware monitoring is not active."""
        for ax in [self.ax_cpu, self.ax_memory, self.ax_disk, self.ax_network, 
                  self.ax_gpu, self.ax_temp, self.ax_process, self.ax_system_info, self.ax_alerts]:
            ax.clear()
            ax.text(0.5, 0.5, 'Hardware monitoring not active\nStart monitoring to see real-time data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_cpu, self.ax_memory, self.ax_disk, self.ax_network, 
                  self.ax_gpu, self.ax_temp, self.ax_process, self.ax_system_info, self.ax_alerts]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()