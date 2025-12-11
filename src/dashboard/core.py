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

try:
    from ..core.logging_config import InstrumentationCollector
    from ..core.models import MemoryTrace
    from ..visualization import (
        PreferenceDriftAnalyzer, 
        AttractorPatternDetector, 
        MemoryNetworkVisualizer,
        SystemHealthDashboard
    )
    from ..monitoring.hardware_monitor import get_hardware_monitor
    from ..monitoring.session_manager import get_session_manager, SessionConfig, SessionState
    from ..monitoring.thread_manager import get_thread_manager
except ImportError:
    # Fallback for direct execution
    from core.logging_config import InstrumentationCollector
    from core.models import MemoryTrace
    from visualization import (
        PreferenceDriftAnalyzer, 
        AttractorPatternDetector, 
        MemoryNetworkVisualizer,
        SystemHealthDashboard
    )
    from hardware_monitor import get_hardware_monitor
    from session_manager import get_session_manager, SessionConfig, SessionState
    from thread_manager import get_thread_manager

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
            self._toggle_monitoring,
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
    
    def _toggle_monitoring(self):
        """Toggle monitoring on/off."""
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()
    
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
                # Update all tabs
                for tab in self.tabs.values():
                    if hasattr(tab, 'update_display'):
                        tab.update_display()
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
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
        """Handle session state changes."""
        self.session_indicator.config(text=f"🔄 Session: {state.value}")
    
    def _on_cycle_complete(self, cycle_num, data):
        """Handle agent cycle completion."""
        self.status_label.config(text=f"Cycle {cycle_num} complete")
    
    def _on_session_error(self, error):
        """Handle session errors."""
        messagebox.showerror("Session Error", str(error))
    
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