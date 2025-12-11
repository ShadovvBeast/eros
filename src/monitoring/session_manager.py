"""
Session Manager for Autonomous Agent

Manages agent sessions with comprehensive control, monitoring, and lifecycle management.
Provides the primary interface for running and controlling agent sessions from the GUI.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import os
from pathlib import Path

try:
    from ..core.agent import AutonomousAgent
    from ..core.config import AgentConfig
    from ..core.logging_config import InstrumentationCollector
    from ..core.models import MemoryTrace
    from .hardware_monitor import get_hardware_monitor
    from .thread_manager import get_thread_manager, register_thread
except ImportError:
    # Fallback for direct execution
    from core.agent import AutonomousAgent
    from core.config import AgentConfig
    from core.logging_config import InstrumentationCollector
    from core.models import MemoryTrace
    from hardware_monitor import get_hardware_monitor
    from thread_manager import get_thread_manager, register_thread


class SessionState(Enum):
    """Session state enumeration"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"


class SessionConfig:
    """Configuration for agent sessions"""
    
    def __init__(self):
        # Session parameters
        self.duration_minutes: float = 10.0
        self.max_cycles: Optional[int] = None
        # Cycle delay removed for maximum efficiency - threading system handles resource management
        
        # Agent configuration
        self.agent_identity: str = "Autonomous Research Agent"
        self.pathos_dimension: int = 128
        self.memory_capacity: int = 1000
        self.exploration_rate: float = 0.3
        
        # Monitoring
        self.enable_hardware_monitoring: bool = True
        self.enable_real_time_visualization: bool = True
        self.save_session_data: bool = True
        
        # Safety limits
        self.max_memory_usage_mb: int = 1024
        self.max_cpu_percent: float = 80.0
        self.emergency_stop_on_error: bool = True
        
        # Output configuration
        self.output_directory: Optional[str] = None
        self.auto_export_interval: int = 100  # cycles
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'duration_minutes': self.duration_minutes,
            'max_cycles': self.max_cycles,
            'agent_identity': self.agent_identity,
            'pathos_dimension': self.pathos_dimension,
            'memory_capacity': self.memory_capacity,
            'exploration_rate': self.exploration_rate,
            'enable_hardware_monitoring': self.enable_hardware_monitoring,
            'enable_real_time_visualization': self.enable_real_time_visualization,
            'save_session_data': self.save_session_data,
            'max_memory_usage_mb': self.max_memory_usage_mb,
            'max_cpu_percent': self.max_cpu_percent,
            'emergency_stop_on_error': self.emergency_stop_on_error,
            'output_directory': self.output_directory,
            'auto_export_interval': self.auto_export_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionConfig':
        """Create from dictionary"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class SessionStats:
    """Session statistics and metrics"""
    
    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_cycles: int = 0
        self.successful_cycles: int = 0
        self.failed_cycles: int = 0
        self.total_tool_calls: int = 0
        self.successful_tool_calls: int = 0
        self.memory_events: int = 0
        self.errors: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
    @property
    def duration(self) -> Optional[timedelta]:
        """Get session duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None
    
    @property
    def cycles_per_minute(self) -> float:
        """Get cycles per minute rate"""
        duration = self.duration
        if duration and duration.total_seconds() > 0:
            return (self.total_cycles / duration.total_seconds()) * 60
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Get cycle success rate"""
        if self.total_cycles > 0:
            return self.successful_cycles / self.total_cycles
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'total_cycles': self.total_cycles,
            'successful_cycles': self.successful_cycles,
            'failed_cycles': self.failed_cycles,
            'total_tool_calls': self.total_tool_calls,
            'successful_tool_calls': self.successful_tool_calls,
            'memory_events': self.memory_events,
            'cycles_per_minute': self.cycles_per_minute,
            'success_rate': self.success_rate,
            'errors': self.errors,
            'performance_metrics': self.performance_metrics
        }


class SessionManager:
    """
    Comprehensive session manager for autonomous agent operations.
    
    Provides:
    - Session lifecycle management
    - Real-time monitoring and control
    - Safety monitoring and emergency stops
    - Performance tracking and optimization
    - Data export and analysis
    """
    
    def __init__(self):
        # Session state
        self.state = SessionState.IDLE
        self.config = SessionConfig()
        self.stats = SessionStats()
        
        # Components
        self.agent: Optional[AutonomousAgent] = None
        self.collector: Optional[InstrumentationCollector] = None
        self.hardware_monitor = get_hardware_monitor()
        
        # Threading
        self.session_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Callbacks
        self.state_change_callbacks: List[Callable[[SessionState], None]] = []
        self.cycle_complete_callbacks: List[Callable[[int, Dict[str, Any]], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        # Safety monitoring
        self.safety_violations: List[Dict[str, Any]] = []
        self.last_safety_check = time.time()
        
        # Output management
        self.session_output_dir: Optional[Path] = None
    
    def add_state_change_callback(self, callback: Callable[[SessionState], None]):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    def add_cycle_complete_callback(self, callback: Callable[[int, Dict[str, Any]], None]):
        """Add callback for cycle completion"""
        self.cycle_complete_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def _set_state(self, new_state: SessionState):
        """Set session state and notify callbacks"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            
            print(f"Session state: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    print(f"State change callback error: {e}")
    
    def configure_session(self, config: SessionConfig):
        """Configure the session"""
        if self.state != SessionState.IDLE:
            raise RuntimeError("Cannot configure session while running")
        
        self.config = config
        
        # Setup output directory
        if config.output_directory:
            self.session_output_dir = Path(config.output_directory)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_output_dir = Path(f"session_output_{timestamp}")
        
        self.session_output_dir.mkdir(exist_ok=True)
        
        print(f"Session configured: {config.duration_minutes} minutes, output: {self.session_output_dir}")
    
    def start_session(self) -> bool:
        """Start the agent session"""
        if self.state != SessionState.IDLE:
            print(f"Cannot start session in state: {self.state}")
            return False
        
        try:
            self._set_state(SessionState.INITIALIZING)
            
            # Initialize components
            self._initialize_session()
            
            # Start monitoring
            if self.config.enable_hardware_monitoring:
                self.hardware_monitor.start_monitoring(1.0)
            
            # Start session thread
            self.stop_event.clear()
            self.pause_event.clear()
            
            self.session_thread = threading.Thread(target=self._session_loop, daemon=True)
            self.session_thread.name = f"AgentSession-{self.stats.session_id[:8]}"
            register_thread(self.session_thread, "session", "SessionManager")
            self.session_thread.start()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.name = f"SessionMonitor-{self.stats.session_id[:8]}"
            register_thread(self.monitoring_thread, "monitoring", "SessionManager")
            self.monitoring_thread.start()
            
            self._set_state(SessionState.RUNNING)
            return True
            
        except Exception as e:
            self._set_state(SessionState.ERROR)
            self._notify_error(e)
            return False
    
    def pause_session(self):
        """Pause the session"""
        if self.state == SessionState.RUNNING:
            self.pause_event.set()
            self._set_state(SessionState.PAUSED)
    
    def resume_session(self):
        """Resume the session"""
        if self.state == SessionState.PAUSED:
            self.pause_event.clear()
            self._set_state(SessionState.RUNNING)
    
    def stop_session(self):
        """Stop the session"""
        if self.state in [SessionState.RUNNING, SessionState.PAUSED]:
            self._set_state(SessionState.STOPPING)
            self.stop_event.set()
            
            # Wait for threads to complete
            if self.session_thread:
                self.session_thread.join(timeout=5)
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2)
            
            # Stop hardware monitoring
            self.hardware_monitor.stop_monitoring()
            
            # Finalize session
            self._finalize_session()
            
            self._set_state(SessionState.COMPLETED)
    
    def emergency_stop(self, reason: str):
        """Emergency stop with reason"""
        print(f"EMERGENCY STOP: {reason}")
        
        self.safety_violations.append({
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'type': 'emergency_stop'
        })
        
        self.stop_session()
    
    def _initialize_session(self):
        """Initialize session components"""
        # Create new stats
        self.stats = SessionStats()
        self.stats.start_time = datetime.now()
        
        # Initialize collector
        self.collector = InstrumentationCollector()
        
        # Initialize full agent with all layers
        agent_config = AgentConfig()
        agent_config.pathos_state_dimension = self.config.pathos_dimension
        agent_config.memory_capacity = self.config.memory_capacity
        
        self.agent = AutonomousAgent(agent_config, instrumentation=self.collector)
        
        # Initialize all layers
        self._initialize_agent_layers()
        
        print(f"Session initialized: {self.stats.session_id}")
    
    def _initialize_agent_layers(self):
        """Initialize all agent layers"""
        try:
            # Import concrete implementations using absolute imports
            from src.logos.logos_layer import LogosLayer
            from src.pathos.pathos_layer import PathosLayer
            from src.memory.memory_system import ConcreteMemorySystem
            from src.ethos.ethos_framework import ConcreteEthosFramework
            from src.tools.tool_layer import ToolLayer
            
            # Create layer configurations
            logos_config = self.agent.config.logos
            pathos_config = self.agent.config.pathos
            memory_config = self.agent.config.memory
            ethos_config = self.agent.config.ethos
            tool_config = self.agent.config.tools
            
            # Initialize layers
            logos = LogosLayer(logos_config)
            pathos = PathosLayer(pathos_config)
            memory = ConcreteMemorySystem(memory_config)
            ethos = ConcreteEthosFramework(ethos_config, pathos)
            tools = ToolLayer(tool_config, ethos)
            
            # Initialize the agent with all layers
            self.agent.initialize_layers(logos, pathos, memory, ethos, tools)
            
            print("✅ All agent layers initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent layers: {e}")
            raise RuntimeError(f"Cannot initialize agent layers: {e}") from e
    
    def _session_loop(self):
        """Main session execution loop"""
        try:
            start_time = time.time()
            cycle = 0
            
            while not self.stop_event.is_set():
                # Check pause - optimized for efficiency
                if self.pause_event.is_set():
                    time.sleep(0.01)  # Minimal pause check delay
                    continue
                
                # Check duration limit
                elapsed_time = time.time() - start_time
                if elapsed_time >= (self.config.duration_minutes * 60):
                    print("Session duration limit reached")
                    break
                
                # Check cycle limit
                if self.config.max_cycles and cycle >= self.config.max_cycles:
                    print("Session cycle limit reached")
                    break
                
                # Execute cycle
                cycle += 1
                cycle_success = self._execute_cycle(cycle)
                
                # Update stats
                self.stats.total_cycles = cycle
                if cycle_success:
                    self.stats.successful_cycles += 1
                else:
                    self.stats.failed_cycles += 1
                
                # Auto-export check
                if (self.config.auto_export_interval > 0 and 
                    cycle % self.config.auto_export_interval == 0):
                    self._auto_export_data(cycle)
                
                # No cycle delay - let threading system manage resources efficiently
            
            print(f"Session loop completed after {cycle} cycles")
            
        except Exception as e:
            print(f"Session loop error: {e}")
            self._notify_error(e)
            self._set_state(SessionState.ERROR)
    
    def _execute_cycle(self, cycle: int) -> bool:
        """Execute a single agent cycle"""
        try:
            cycle_start = time.time()
            
            # Execute real agent cycle - no simulation fallback
            if not self.agent or not hasattr(self.agent, 'run_cycle'):
                raise RuntimeError("Agent not properly initialized - cannot execute cycles")
            
            cycle_result = self.agent.run_cycle()
            
            # Update stats from real agent results
            if cycle_result.get('tool_used'):
                self.stats.total_tool_calls += 1
                if cycle_result.get('external_reward', 0) > 0:
                    self.stats.successful_tool_calls += 1
            
            if cycle_result.get('memory_stored', False):
                self.stats.memory_events += 1
            
            cycle_duration = time.time() - cycle_start
            
            # Notify cycle completion
            cycle_data = {
                'cycle': cycle,
                'duration': cycle_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            for callback in self.cycle_complete_callbacks:
                try:
                    callback(cycle, cycle_data)
                except Exception as e:
                    print(f"Cycle callback error: {e}")
            
            return True
            
        except Exception as e:
            print(f"Cycle {cycle} error: {e}")
            self.stats.errors.append({
                'cycle': cycle,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            if self.config.emergency_stop_on_error:
                self.emergency_stop(f"Cycle error: {e}")
            
            return False
    

    
    def _monitoring_loop(self):
        """Safety and performance monitoring loop"""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Safety checks every 2 seconds - more responsive
                if current_time - self.last_safety_check >= 2.0:
                    self._perform_safety_checks()
                    self.last_safety_check = current_time
                
                time.sleep(0.5)  # Faster monitoring for efficiency
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _perform_safety_checks(self):
        """Perform safety and resource checks"""
        try:
            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.max_memory_usage_mb:
                self.emergency_stop(f"Memory usage exceeded: {memory_mb:.1f}MB > {self.config.max_memory_usage_mb}MB")
                return
            
            # Check CPU usage
            cpu_percent = process.cpu_percent()
            if cpu_percent > self.config.max_cpu_percent:
                self.safety_violations.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'high_cpu',
                    'value': cpu_percent,
                    'limit': self.config.max_cpu_percent
                })
            
            # Update performance metrics
            self.stats.performance_metrics.update({
                'memory_usage_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'cycles_per_minute': self.stats.cycles_per_minute
            })
            
        except Exception as e:
            print(f"Safety check error: {e}")
    
    def _auto_export_data(self, cycle: int):
        """Auto-export session data"""
        try:
            if self.session_output_dir:
                export_file = self.session_output_dir / f"auto_export_cycle_{cycle}.json"
                self.export_session_data(str(export_file))
                print(f"Auto-exported data at cycle {cycle}")
        except Exception as e:
            print(f"Auto-export error: {e}")
    
    def _finalize_session(self):
        """Finalize session and save data"""
        self.stats.end_time = datetime.now()
        
        if self.config.save_session_data and self.session_output_dir:
            # Save final session data
            final_export = self.session_output_dir / "final_session_data.json"
            self.export_session_data(str(final_export))
            
            # Save hardware metrics
            if self.config.enable_hardware_monitoring:
                hardware_export = self.session_output_dir / "hardware_metrics.json"
                self.hardware_monitor.export_metrics(str(hardware_export))
            
            print(f"Session data saved to: {self.session_output_dir}")
    
    def _notify_error(self, error: Exception):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                print(f"Error callback failed: {e}")
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status"""
        return {
            'state': self.state.value,
            'stats': self.stats.to_dict(),
            'config': self.config.to_dict(),
            'safety_violations': self.safety_violations,
            'output_directory': str(self.session_output_dir) if self.session_output_dir else None
        }
    
    def export_session_data(self, filepath: str) -> bool:
        """Export complete session data"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'session_status': self.get_session_status(),
                'collector_metrics': self.collector.get_metrics_summary() if self.collector else {},
                'hardware_summary': self.hardware_monitor.get_current_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Export error: {e}")
            return False


# Global session manager instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager