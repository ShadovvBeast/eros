"""
Core Interactive Dashboard

Main dashboard class and setup functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import json
import os

# Import with fallback for different execution contexts
try:
    # Try relative imports first (when imported as module)
    from ..core.logging_config import InstrumentationCollector
    from ..core.models import MemoryTrace
    from ..monitoring.hardware_monitor import get_hardware_monitor
    from ..monitoring.session_manager import get_session_manager, SessionConfig, SessionState
    from ..monitoring.thread_manager import get_thread_manager
except ImportError:
    try:
        # Try absolute imports (when run from main.py)
        from core.logging_config import InstrumentationCollector
        from core.models import MemoryTrace
        from monitoring.hardware_monitor import get_hardware_monitor
        from monitoring.session_manager import get_session_manager, SessionConfig, SessionState
        from monitoring.thread_manager import get_thread_manager
    except ImportError:
        # Create mock implementations for testing
        print("⚠️  Using mock implementations for dashboard components")
        
        class InstrumentationCollector:
            def __init__(self):
                self.metrics = {}
            def get_metrics_summary(self):
                return {}
        
        class MemoryTrace:
            def __init__(self):
                pass
        
        def get_hardware_monitor():
            class MockHardwareMonitor:
                def __init__(self):
                    self.is_monitoring = False
                def get_current_summary(self):
                    return {'current_metrics': {}, 'system_info': {}}
            return MockHardwareMonitor()
        
        def get_session_manager():
            class MockSessionManager:
                def __init__(self):
                    from enum import Enum
                    class SessionState(Enum):
                        IDLE = "idle"
                        RUNNING = "running"
                    self.state = SessionState.IDLE
                def add_state_change_callback(self, callback): pass
                def add_cycle_complete_callback(self, callback): pass
                def add_error_callback(self, callback): pass
            return MockSessionManager()
        
        def get_thread_manager():
            class MockThreadManager:
                def get_all_threads(self): return []
                def get_active_threads(self): return []
            return MockThreadManager()
        
        class SessionConfig:
            def __init__(self):
                self.duration_minutes = 5.0
                self.agent_identity = "Test Agent"
                self.pathos_dimension = 128
                self.exploration_rate = 0.1
                self.output_directory = "test_output"
        
        class SessionState:
            IDLE = "idle"
            RUNNING = "running"
            COMPLETED = "completed"
            ERROR = "error"

from .session_controls import SessionControlPanel
from .log_system import LogAuditSystem
from .tabs import (
    HealthTab, HardwareTab, PathosTab, MemoryTab,
    PreferenceTab, AttractorTab, PerformanceTab,
    ThreadManagementTab, LogAuditTab
)


class InteractiveDashboard:
    """
    Unified interactive dashboard for autonomous agent monitoring.
    
    Provides a comprehensive, tabbed interface for real-time monitoring
    and analysis of all agent systems and components.
    """
    
    def __init__(self, collector: Optional[InstrumentationCollector] = None, 
                 pathos_states: Optional[List] = None,
                 memory_traces: Optional[List[MemoryTrace]] = None,
                 update_interval: float = 1.0):
        """Initialize the interactive dashboard."""
        
        # Core components
        self.collector = collector or InstrumentationCollector()
        self.pathos_states = pathos_states or []
        self.memory_traces = memory_traces or []
        self.update_interval = update_interval
        
        # System managers
        self.session_manager = get_session_manager()
        self.hardware_monitor = get_hardware_monitor()
        self.thread_manager = get_thread_manager()
        
        # Dashboard state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.canvases = {}
        self.figures = {}
        
        # Session configuration
        self.session_config = SessionConfig()
        self.session_config.duration_minutes = 5.0
        self.session_config.agent_identity = "Autonomous Agent"
        self.session_config.pathos_dimension = 128
        self.session_config.exploration_rate = 0.1
        self.session_config.output_directory = "session_output"
        
        # Initialize components
        self.session_controls = None
        self.log_system = None
        self.tabs = {}
        
        # Setup GUI
        self._setup_gui()
        self._setup_session_callbacks()
        
        # Start monitoring automatically (for hardware data, etc.)
        self._start_monitoring()
    
    def _setup_gui(self):
        """Setup the main GUI interface."""
        self.root = tk.Tk()
        self.root.title("🎯 E.R.O.S Control Center")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self._create_header(main_frame)
        
        # Session controls
        self.session_controls = SessionControlPanel(
            main_frame, 
            self.session_manager,
            self.session_config,
            self._export_data,
            self._open_agent_config
        )
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create all tabs
        self._create_tabs()
        
        # Status bar
        self._create_status_bar(main_frame)
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_session_callbacks(self):
        """Setup session manager callbacks."""
        self.session_manager.add_state_change_callback(self._on_session_state_change)
        self.session_manager.add_cycle_complete_callback(self._on_cycle_complete)
        self.session_manager.add_error_callback(self._on_session_error)
    
    def _create_header(self, parent):
        """Create header with title and session info."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="🎯 E.R.O.S Control Center",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Session info
        session_info_label = ttk.Label(
            header_frame,
            text=f"Session: {self.session_manager.state.value} | Hardware: Active",
            font=('Arial', 10)
        )
        session_info_label.grid(row=0, column=1, sticky=tk.E)
        header_frame.columnconfigure(1, weight=1)
    
    def _create_tabs(self):
        """Create all dashboard tabs."""
        # Initialize tab components
        self.tabs['health'] = HealthTab(self.notebook, self.collector, self.hardware_monitor)
        self.tabs['hardware'] = HardwareTab(self.notebook, self.hardware_monitor)
        self.tabs['pathos'] = PathosTab(self.notebook, self.collector, self.pathos_states)
        self.tabs['memory'] = MemoryTab(self.notebook, self.memory_traces)
        self.tabs['preference'] = PreferenceTab(self.notebook, self.collector)
        self.tabs['attractor'] = AttractorTab(self.notebook, self.collector)
        self.tabs['performance'] = PerformanceTab(self.notebook, self.collector)
        self.tabs['threads'] = ThreadManagementTab(self.notebook, self.thread_manager)
        
        # Log system is special - needs more integration
        self.log_system = LogAuditSystem(self.notebook, self.session_manager)
        self.tabs['logs'] = self.log_system
    
    def _create_status_bar(self, parent):
        """Create comprehensive status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status indicators
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # Monitoring indicator
        self.monitoring_indicator = ttk.Label(
            status_frame, 
            text="⚫ Monitoring: Stopped",
            foreground="red"
        )
        self.monitoring_indicator.pack(side=tk.LEFT, padx=(20, 0))
        
        # Session indicator
        self.session_indicator = ttk.Label(
            status_frame,
            text=f"🔄 Session: {self.session_manager.state.value}",
            foreground="blue"
        )
        self.session_indicator.pack(side=tk.LEFT, padx=(20, 0))
        
        # Hardware indicator
        hardware_status = "🟢 Online" if self.hardware_monitor.is_monitoring else "🔴 Offline"
        hardware_indicator = ttk.Label(
            status_frame,
            text=f"🖥️ Hardware: {hardware_status}",
            foreground="green" if self.hardware_monitor.is_monitoring else "red"
        )
        hardware_indicator.pack(side=tk.RIGHT)
    

    
    def _start_monitoring(self):
        """Start monitoring."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Update UI
        self.monitoring_indicator.config(
            text="🟢 Monitoring: Active",
            foreground="green"
        )
        
        # Start log system
        if self.log_system:
            self.log_system.start_monitoring()
    
    def _stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        
        # Update UI
        self.monitoring_indicator.config(
            text="⚫ Monitoring: Stopped",
            foreground="red"
        )
        
        # Stop log system
        if self.log_system:
            self.log_system.stop_monitoring()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Refresh data connections if session is running
                if self.session_manager.state == SessionState.RUNNING:
                    self._refresh_data_connections()
                
                # Update all tabs
                for tab in self.tabs.values():
                    if hasattr(tab, 'update_display'):
                        tab.update_display()
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def _refresh_data_connections(self):
        """Refresh data connections to session manager."""
        # Update collector if it changed
        session_collector = self.session_manager.get_current_collector()
        if session_collector and session_collector != self.collector:
            self.collector = session_collector
            self._update_tabs_collector()
            print("🔄 Dashboard collector updated")
        
        # Debug: Print collector status every 10 updates
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 1
        
        if self._debug_counter % 10 == 0:
            self._print_collector_debug()
        
        # Update memory traces if they changed
        session_agent = self.session_manager.get_current_agent()
        if session_agent and hasattr(session_agent, 'memory'):
            if hasattr(session_agent.memory, 'traces'):
                if session_agent.memory.traces != self.memory_traces:
                    self.memory_traces = session_agent.memory.traces
                    if 'memory' in self.tabs:
                        self.tabs['memory'].memory_traces = self.memory_traces
    
    def _print_collector_debug(self):
        """Print debug information about collector status."""
        if hasattr(self.collector, 'metrics'):
            metrics = self.collector.metrics
            total_data = sum(len(metrics[key]) if isinstance(metrics[key], list) else 0 for key in metrics)
            print(f"🔍 Collector debug: {total_data} total data points")
            for key, value in metrics.items():
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)} items")
        else:
            print("🔍 Collector debug: No metrics attribute")
    
    def _export_data(self):
        """Export current data and visualizations."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"dashboard_export_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            # Export from each tab
            for name, tab in self.tabs.items():
                if hasattr(tab, 'export_data'):
                    tab.export_data(export_dir)
            
            messagebox.showinfo("Export Complete", f"Data exported to {export_dir}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def _open_agent_config(self):
        """Open agent configuration dialog."""
        # This will be implemented by the session controls
        if self.session_controls:
            self.session_controls.open_config_dialog()
    
    def _on_session_state_change(self, state):
        """Handle session state changes and automatically manage monitoring."""
        self.session_indicator.config(text=f"🔄 Session: {state.value}")
        
        # Update collector reference when session starts
        if state == SessionState.RUNNING:
            # Use the session manager's collector for real data
            session_collector = self.session_manager.get_current_collector()
            if session_collector:
                self.collector = session_collector
                # Update all tabs with the new collector
                self._update_tabs_collector()
                print("✅ Dashboard connected to session collector")
            
            # Update memory traces from agent if available
            session_agent = self.session_manager.get_current_agent()
            if session_agent and hasattr(session_agent, 'memory'):
                if hasattr(session_agent.memory, 'traces'):
                    self.memory_traces = session_agent.memory.traces
                    # Update memory tab
                    if 'memory' in self.tabs:
                        self.tabs['memory'].memory_traces = self.memory_traces
                        print("✅ Dashboard connected to agent memory traces")
            
            if not self.is_monitoring:
                self._start_monitoring()
        
        # Automatically stop monitoring when session ends
        elif state in [SessionState.COMPLETED, SessionState.ERROR, SessionState.IDLE] and self.is_monitoring:
            self._stop_monitoring()
    
    def _on_cycle_complete(self, cycle_num, data):
        """Handle agent cycle completion."""
        self.status_label.config(text=f"Cycle {cycle_num} complete")
    
    def _on_session_error(self, error):
        """Handle session errors."""
        messagebox.showerror("Session Error", str(error))
    
    def _update_tabs_collector(self):
        """Update all tabs with the current collector."""
        # Update tabs that use the collector
        if 'health' in self.tabs:
            self.tabs['health'].collector = self.collector
        if 'pathos' in self.tabs:
            self.tabs['pathos'].collector = self.collector
        if 'preference' in self.tabs:
            self.tabs['preference'].collector = self.collector
        if 'attractor' in self.tabs:
            self.tabs['attractor'].collector = self.collector
        if 'performance' in self.tabs:
            self.tabs['performance'].collector = self.collector
    
    def _on_closing(self):
        """Handle window closing."""
        self._stop_monitoring()
        if self.session_manager.state != SessionState.IDLE:
            if messagebox.askyesno("Confirm Exit", "Session is active. Stop and exit?"):
                self.session_manager.stop_session()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run the dashboard."""
        self.root.mainloop()


def create_interactive_dashboard(collector=None, update_interval=1.0):
    """Create and return an interactive dashboard instance."""
    return InteractiveDashboard(
        collector=collector,
        update_interval=update_interval
    )