"""
Hardware Monitoring Module

Comprehensive system hardware monitoring for the autonomous agent dashboard.
Tracks CPU, memory, disk, network, GPU, and other system resources.
"""

import psutil
import platform
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import json

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False


class HardwareMonitor:
    """
    Comprehensive hardware monitoring system.
    
    Monitors:
    - CPU usage, frequency, temperature
    - Memory usage (RAM, swap)
    - Disk usage, I/O operations
    - Network I/O
    - GPU usage and memory (if available)
    - System temperatures
    - Battery status (if applicable)
    - Process-specific metrics
    """
    
    def __init__(self, history_length: int = 300):
        """
        Initialize hardware monitor.
        
        Args:
            history_length: Number of data points to keep in history
        """
        self.history_length = history_length
        self.is_monitoring = False
        self.monitor_thread = None
        self.update_interval = 0.5  # seconds - optimized for efficiency
        
        # Data storage
        self.cpu_history = deque(maxlen=history_length)
        self.memory_history = deque(maxlen=history_length)
        self.disk_history = deque(maxlen=history_length)
        self.network_history = deque(maxlen=history_length)
        self.gpu_history = deque(maxlen=history_length)
        self.temperature_history = deque(maxlen=history_length)
        self.process_history = deque(maxlen=history_length)
        
        # System info
        self.system_info = self._get_system_info()
        
        # Initialize GPU monitoring if available
        self.gpu_available = GPU_AVAILABLE
        self.nvidia_ml_available = NVIDIA_ML_AVAILABLE
        
        if self.nvidia_ml_available:
            try:
                nvml.nvmlInit()
            except:
                self.nvidia_ml_available = False
        
        # Process tracking
        self.current_process = psutil.Process()
        
        # Network baseline
        self.network_baseline = psutil.net_io_counters()
        self.last_network_time = time.time()
        
        # Disk baseline
        self.disk_baseline = psutil.disk_io_counters()
        self.last_disk_time = time.time()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information"""
        try:
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            
            info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq_max': cpu_freq.max if cpu_freq else None,
                'memory_total': memory.total,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'python_version': platform.python_version(),
            }
            
            # Disk information
            disk_partitions = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_partitions.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free
                    })
                except PermissionError:
                    continue
            
            info['disk_partitions'] = disk_partitions
            
            # GPU information
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    info['gpus'] = [
                        {
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_total': gpu.memoryTotal,
                            'driver': gpu.driver
                        }
                        for gpu in gpus
                    ]
                except:
                    info['gpus'] = []
            else:
                info['gpus'] = []
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def start_monitoring(self, update_interval: float = 1.0):
        """Start hardware monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.update_interval = update_interval
        self.is_monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.name = "HardwareMonitor"
        
        # Register with thread manager
        try:
            from .thread_manager import register_thread
            register_thread(self.monitor_thread, "hardware", "HardwareMonitor")
        except ImportError:
            pass  # Thread manager not available
        
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                timestamp = datetime.now()
                
                # Collect all metrics
                cpu_data = self._get_cpu_metrics(timestamp)
                memory_data = self._get_memory_metrics(timestamp)
                disk_data = self._get_disk_metrics(timestamp)
                network_data = self._get_network_metrics(timestamp)
                gpu_data = self._get_gpu_metrics(timestamp)
                temperature_data = self._get_temperature_metrics(timestamp)
                process_data = self._get_process_metrics(timestamp)
                
                # Store in history
                self.cpu_history.append(cpu_data)
                self.memory_history.append(memory_data)
                self.disk_history.append(disk_data)
                self.network_history.append(network_data)
                self.gpu_history.append(gpu_data)
                self.temperature_history.append(temperature_data)
                self.process_history.append(process_data)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Hardware monitoring error: {e}")
                time.sleep(1)
    
    def _get_cpu_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Get CPU metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            cpu_freq = psutil.cpu_freq()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            
            return {
                'timestamp': timestamp.isoformat(),
                'cpu_percent': cpu_percent,
                'cpu_per_core': cpu_per_core,
                'cpu_freq_current': cpu_freq.current if cpu_freq else None,
                'cpu_freq_min': cpu_freq.min if cpu_freq else None,
                'cpu_freq_max': cpu_freq.max if cpu_freq else None,
                'load_avg': load_avg,
                'cpu_count': psutil.cpu_count()
            }
        except Exception as e:
            return {'timestamp': timestamp.isoformat(), 'error': str(e)}
    
    def _get_memory_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Get memory metrics"""
        try:
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            return {
                'timestamp': timestamp.isoformat(),
                'virtual_total': virtual_memory.total,
                'virtual_available': virtual_memory.available,
                'virtual_used': virtual_memory.used,
                'virtual_percent': virtual_memory.percent,
                'virtual_free': virtual_memory.free,
                'swap_total': swap_memory.total,
                'swap_used': swap_memory.used,
                'swap_free': swap_memory.free,
                'swap_percent': swap_memory.percent
            }
        except Exception as e:
            return {'timestamp': timestamp.isoformat(), 'error': str(e)}
    
    def _get_disk_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Get disk metrics"""
        try:
            current_time = time.time()
            disk_io = psutil.disk_io_counters()
            
            # Calculate rates
            time_delta = current_time - self.last_disk_time
            read_rate = 0
            write_rate = 0
            
            if time_delta > 0 and self.disk_baseline:
                read_rate = (disk_io.read_bytes - self.disk_baseline.read_bytes) / time_delta
                write_rate = (disk_io.write_bytes - self.disk_baseline.write_bytes) / time_delta
            
            # Update baseline
            self.disk_baseline = disk_io
            self.last_disk_time = current_time
            
            # Disk usage for main partitions
            disk_usage = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100
                    })
                except PermissionError:
                    continue
            
            return {
                'timestamp': timestamp.isoformat(),
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_rate': read_rate,
                'write_rate': write_rate,
                'disk_usage': disk_usage
            }
        except Exception as e:
            return {'timestamp': timestamp.isoformat(), 'error': str(e)}
    
    def _get_network_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Get network metrics"""
        try:
            current_time = time.time()
            net_io = psutil.net_io_counters()
            
            # Calculate rates
            time_delta = current_time - self.last_network_time
            bytes_sent_rate = 0
            bytes_recv_rate = 0
            
            if time_delta > 0 and self.network_baseline:
                bytes_sent_rate = (net_io.bytes_sent - self.network_baseline.bytes_sent) / time_delta
                bytes_recv_rate = (net_io.bytes_recv - self.network_baseline.bytes_recv) / time_delta
            
            # Update baseline
            self.network_baseline = net_io
            self.last_network_time = current_time
            
            return {
                'timestamp': timestamp.isoformat(),
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'bytes_sent_rate': bytes_sent_rate,
                'bytes_recv_rate': bytes_recv_rate
            }
        except Exception as e:
            return {'timestamp': timestamp.isoformat(), 'error': str(e)}
    
    def _get_gpu_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Get GPU metrics"""
        try:
            gpu_data = {
                'timestamp': timestamp.isoformat(),
                'gpus': []
            }
            
            if self.gpu_available:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_info = {
                            'id': gpu.id,
                            'name': gpu.name,
                            'load': gpu.load * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            'temperature': gpu.temperature
                        }
                        gpu_data['gpus'].append(gpu_info)
                except Exception as e:
                    gpu_data['error'] = str(e)
            
            # Additional NVIDIA-specific metrics
            if self.nvidia_ml_available:
                try:
                    device_count = nvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # Power usage
                        try:
                            power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                            if i < len(gpu_data['gpus']):
                                gpu_data['gpus'][i]['power_usage'] = power
                        except:
                            pass
                        
                        # Clock speeds
                        try:
                            graphics_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                            memory_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                            if i < len(gpu_data['gpus']):
                                gpu_data['gpus'][i]['graphics_clock'] = graphics_clock
                                gpu_data['gpus'][i]['memory_clock'] = memory_clock
                        except:
                            pass
                            
                except Exception as e:
                    gpu_data['nvidia_error'] = str(e)
            
            return gpu_data
            
        except Exception as e:
            return {'timestamp': timestamp.isoformat(), 'error': str(e)}
    
    def _get_temperature_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Get temperature metrics"""
        try:
            temperatures = {}
            
            # System temperatures (if available)
            if hasattr(psutil, 'sensors_temperatures'):
                temp_sensors = psutil.sensors_temperatures()
                for name, entries in temp_sensors.items():
                    temperatures[name] = []
                    for entry in entries:
                        temperatures[name].append({
                            'label': entry.label or 'Unknown',
                            'current': entry.current,
                            'high': entry.high,
                            'critical': entry.critical
                        })
            
            return {
                'timestamp': timestamp.isoformat(),
                'temperatures': temperatures
            }
        except Exception as e:
            return {'timestamp': timestamp.isoformat(), 'error': str(e)}
    
    def _get_process_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Get process-specific metrics"""
        try:
            # Current process metrics
            process_info = {
                'timestamp': timestamp.isoformat(),
                'pid': self.current_process.pid,
                'cpu_percent': self.current_process.cpu_percent(),
                'memory_info': self.current_process.memory_info()._asdict(),
                'memory_percent': self.current_process.memory_percent(),
                'num_threads': self.current_process.num_threads(),
                'create_time': self.current_process.create_time(),
                'status': self.current_process.status()
            }
            
            # System-wide process stats
            process_info['system_processes'] = len(psutil.pids())
            
            # Top processes by CPU
            top_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if proc.info['cpu_percent'] > 0:
                        top_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage and take top 5
            top_processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            process_info['top_cpu_processes'] = top_processes[:5]
            
            return process_info
            
        except Exception as e:
            return {'timestamp': timestamp.isoformat(), 'error': str(e)}
    
    def get_current_summary(self) -> Dict[str, Any]:
        """Get current hardware summary"""
        try:
            timestamp = datetime.now()
            
            summary = {
                'timestamp': timestamp.isoformat(),
                'system_info': self.system_info,
                'current_metrics': {
                    'cpu': self._get_cpu_metrics(timestamp),
                    'memory': self._get_memory_metrics(timestamp),
                    'disk': self._get_disk_metrics(timestamp),
                    'network': self._get_network_metrics(timestamp),
                    'gpu': self._get_gpu_metrics(timestamp),
                    'temperature': self._get_temperature_metrics(timestamp),
                    'process': self._get_process_metrics(timestamp)
                }
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics (compatibility alias for get_current_summary)"""
        summary = self.get_current_summary()
        return summary.get('current_metrics', {})
    
    def get_history_data(self, metric_type: str, duration_minutes: int = 5) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric type"""
        history_map = {
            'cpu': self.cpu_history,
            'memory': self.memory_history,
            'disk': self.disk_history,
            'network': self.network_history,
            'gpu': self.gpu_history,
            'temperature': self.temperature_history,
            'process': self.process_history
        }
        
        if metric_type not in history_map:
            return []
        
        history = history_map[metric_type]
        
        # Filter by duration
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        filtered_history = []
        
        for entry in history:
            try:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_time:
                    filtered_history.append(entry)
            except:
                continue
        
        return filtered_history
    
    def export_metrics(self, filepath: str):
        """Export all current metrics to JSON file"""
        try:
            data = {
                'export_time': datetime.now().isoformat(),
                'system_info': self.system_info,
                'current_summary': self.get_current_summary(),
                'history': {
                    'cpu': list(self.cpu_history),
                    'memory': list(self.memory_history),
                    'disk': list(self.disk_history),
                    'network': list(self.network_history),
                    'gpu': list(self.gpu_history),
                    'temperature': list(self.temperature_history),
                    'process': list(self.process_history)
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            print(f"Export error: {e}")
            return False


# Singleton instance for global access
_hardware_monitor = None

def get_hardware_monitor() -> HardwareMonitor:
    """Get global hardware monitor instance"""
    global _hardware_monitor
    if _hardware_monitor is None:
        _hardware_monitor = HardwareMonitor()
    return _hardware_monitor