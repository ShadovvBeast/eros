"""
Interactive Dashboard for Autonomous Agent Monitoring

A unified, tabbed interface for real-time monitoring and analysis of the autonomous agent system.
Provides comprehensive visualization capabilities in a single, professional-looking interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import json
import os

try:
    from .logging_config import InstrumentationCollector
    from .models import MemoryTrace
    from .visualization import (
        PreferenceDriftAnalyzer, 
        AttractorPatternDetector, 
        MemoryNetworkVisualizer,
        SystemHealthDashboard
    )
    from .hardware_monitor import get_hardware_monitor
    from .session_manager import get_session_manager, SessionConfig, SessionState
except ImportError:
    # Fallback for direct execution
    from logging_config import InstrumentationCollector
    from models import MemoryTrace
    from visualization import (
        PreferenceDriftAnalyzer, 
        AttractorPatternDetector, 
        MemoryNetworkVisualizer,
        SystemHealthDashboard
    )
    from hardware_monitor import get_hardware_monitor
    from session_manager import get_session_manager, SessionConfig, SessionState


class InteractiveDashboard:
    """
    Unified interactive dashboard for autonomous agent monitoring.
    
    Features:
    - Real-time system health monitoring
    - Pathos state visualization
    - Memory network analysis
    - Preference drift tracking
    - Attractor pattern detection
    - Performance metrics
    """
    
    def __init__(self, collector: Optional[InstrumentationCollector] = None, 
                 memory_traces: Optional[List[MemoryTrace]] = None,
                 update_interval: float = 2.0):
        """
        Initialize the interactive dashboard.
        
        Args:
            collector: InstrumentationCollector for monitoring data (optional)
            memory_traces: Optional list of memory traces
            update_interval: Update frequency in seconds
        """
        self.collector = collector
        self.memory_traces = memory_traces or []
        self.update_interval = update_interval
        
        # Initialize analyzers
        self.drift_analyzer = PreferenceDriftAnalyzer()
        self.attractor_detector = AttractorPatternDetector()
        self.memory_visualizer = MemoryNetworkVisualizer()
        self.health_dashboard = SystemHealthDashboard()
        
        # Hardware and session management
        self.hardware_monitor = get_hardware_monitor()
        self.session_manager = get_session_manager()
        
        # GUI components
        self.root = None
        self.notebook = None
        self.tabs = {}
        self.figures = {}
        self.canvases = {}
        
        # Real-time data
        self.is_monitoring = False
        self.update_thread = None
        
        # State tracking
        self.pathos_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        self.hardware_history = deque(maxlen=300)
        
        # Session state
        self.session_config = SessionConfig()
        
        # Setup callbacks
        self._setup_session_callbacks()
        
        # Initialize log system early
        self.log_buffer = deque(maxlen=1000)
        self.event_counts = {
            'total': 0, 'session': 0, 'tool': 0, 'error': 0,
            'thread': 0, 'hardware': 0, 'memory': 0, 'pathos': 0, 'ethos': 0
        }
        self.last_event_time = None
        self.event_rate_history = deque(maxlen=60)
        
        # Setup GUI
        self._setup_gui()
    
    def _setup_gui(self):
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("Autonomous Agent Control Center")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors and fonts
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Status.TLabel', font=('Arial', 10))
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Warning.TLabel', font=('Arial', 10), foreground='#f39c12')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title and session info
        self._create_header(main_frame)
        
        # Session control panel
        self._create_session_controls(main_frame)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create tabs
        self._create_tabs()
        
        # Status bar
        self._create_status_bar(main_frame)
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_session_callbacks(self):
        """Setup session manager callbacks"""
        self.session_manager.add_state_change_callback(self._on_session_state_change)
        self.session_manager.add_cycle_complete_callback(self._on_cycle_complete)
        self.session_manager.add_error_callback(self._on_session_error)
    
    def _create_header(self, parent):
        """Create header with title and session info"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(header_frame, text="🤖 Autonomous Agent Control Center", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Session info
        self.session_info_var = tk.StringVar(value="No active session")
        session_info_label = ttk.Label(header_frame, textvariable=self.session_info_var,
                                      style='Header.TLabel')
        session_info_label.grid(row=0, column=1, sticky=tk.E)
    
    def _create_session_controls(self, parent):
        """Create comprehensive session control panel"""
        control_frame = ttk.LabelFrame(parent, text="Session Control", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(2, weight=1)
        
        # Session configuration
        config_frame = ttk.Frame(control_frame)
        config_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Duration setting
        ttk.Label(config_frame, text="Duration (min):").grid(row=0, column=0, padx=(0, 5))
        self.duration_var = tk.DoubleVar(value=self.session_config.duration_minutes)
        duration_spinbox = ttk.Spinbox(config_frame, from_=0.5, to=60.0, increment=0.5,
                                      width=8, textvariable=self.duration_var)
        duration_spinbox.grid(row=0, column=1, padx=(0, 15))
        
        # Max cycles setting
        ttk.Label(config_frame, text="Max Cycles:").grid(row=0, column=2, padx=(0, 5))
        self.max_cycles_var = tk.StringVar(value="")
        max_cycles_entry = ttk.Entry(config_frame, width=8, textvariable=self.max_cycles_var)
        max_cycles_entry.grid(row=0, column=3, padx=(0, 15))
        
        # Cycle delay setting
        ttk.Label(config_frame, text="Cycle Delay (s):").grid(row=0, column=4, padx=(0, 5))
        self.cycle_delay_var = tk.DoubleVar(value=self.session_config.cycle_delay)
        cycle_delay_spinbox = ttk.Spinbox(config_frame, from_=0.0, to=2.0, increment=0.1,
                                         width=8, textvariable=self.cycle_delay_var)
        cycle_delay_spinbox.grid(row=0, column=5, padx=(0, 15))
        
        # Agent configuration button
        config_button = ttk.Button(config_frame, text="⚙️ Configure Agent", 
                                  command=self._open_agent_config)
        config_button.grid(row=0, column=6)
        
        # Session control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Start/Stop session button
        self.session_button = ttk.Button(button_frame, text="▶️ Start Session", 
                                        command=self._toggle_session, width=15)
        self.session_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Pause/Resume button
        self.pause_button = ttk.Button(button_frame, text="⏸️ Pause", 
                                      command=self._toggle_pause, width=12, state='disabled')
        self.pause_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Emergency stop button
        emergency_button = ttk.Button(button_frame, text="🛑 Emergency Stop", 
                                     command=self._emergency_stop, width=15)
        emergency_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Monitoring controls
        monitor_frame = ttk.Frame(button_frame)
        monitor_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        # Start/Stop monitoring button
        self.monitor_button = ttk.Button(monitor_frame, text="📊 Start Monitoring", 
                                        command=self._toggle_monitoring, width=15)
        self.monitor_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Update interval
        ttk.Label(monitor_frame, text="Update (s):").pack(side=tk.LEFT, padx=(0, 5))
        self.interval_var = tk.DoubleVar(value=self.update_interval)
        interval_spinbox = ttk.Spinbox(monitor_frame, from_=0.5, to=10.0, increment=0.5,
                                      width=6, textvariable=self.interval_var)
        interval_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        # Export controls
        export_frame = ttk.Frame(button_frame)
        export_frame.pack(side=tk.RIGHT)
        
        # Export button
        export_button = ttk.Button(export_frame, text="💾 Export Data", 
                                  command=self._export_data, width=12)
        export_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Refresh button
        refresh_button = ttk.Button(export_frame, text="🔄 Refresh", 
                                   command=self._manual_update, width=10)
        refresh_button.pack(side=tk.LEFT)
    
    def _create_tabs(self):
        """Create all dashboard tabs"""
        # Tab 1: System Health Overview (Enhanced with Hardware)
        self._create_health_tab()
        
        # Tab 2: Hardware Monitoring
        self._create_hardware_tab()
        
        # Tab 3: Real-time Pathos State
        self._create_pathos_tab()
        
        # Tab 4: Memory Network
        self._create_memory_tab()
        
        # Tab 5: Preference Drift Analysis
        self._create_preference_tab()
        
        # Tab 6: Attractor Patterns
        self._create_attractor_tab()
        
        # Tab 7: Performance Metrics
        self._create_performance_tab()
        
        # Tab 8: Thread Management
        self._create_thread_management_tab()
        
        # Tab 9: Log & Audit Trail
        self._create_log_audit_tab()
    
    def _create_health_tab(self):
        """Create enhanced system health overview tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🏥 System Health")
        
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        fig.suptitle('System Health Dashboard', fontsize=16, fontweight='bold')
        
        # Create subplots grid (4x4 for more comprehensive view)
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
        
        # Store axes for updates
        self.health_axes = {
            'overview': fig.add_subplot(gs[0, 0]),
            'performance': fig.add_subplot(gs[0, 1]),
            'hardware_summary': fig.add_subplot(gs[0, 2]),
            'session_status': fig.add_subplot(gs[0, 3]),
            'memory': fig.add_subplot(gs[1, 0]),
            'tools': fig.add_subplot(gs[1, 1]),
            'errors': fig.add_subplot(gs[1, 2]),
            'cycles': fig.add_subplot(gs[1, 3]),
            'cpu_memory': fig.add_subplot(gs[2, 0]),
            'disk_network': fig.add_subplot(gs[2, 1]),
            'preferences': fig.add_subplot(gs[2, 2]),
            'behavior': fig.add_subplot(gs[2, 3]),
            'health_scores': fig.add_subplot(gs[3, 0:2]),
            'alerts': fig.add_subplot(gs[3, 2:4])
        }
        
        self.figures['health'] = fig
        self._add_canvas_to_frame(fig, frame, 'health')
    
    def _create_hardware_tab(self):
        """Create comprehensive hardware monitoring tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🖥️ Hardware Monitor")
        
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        fig.suptitle('Hardware Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # Create subplots grid
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # Store axes for updates
        self.hardware_axes = {
            'cpu_usage': fig.add_subplot(gs[0, 0]),
            'memory_usage': fig.add_subplot(gs[0, 1]),
            'disk_usage': fig.add_subplot(gs[0, 2]),
            'cpu_cores': fig.add_subplot(gs[1, 0]),
            'memory_details': fig.add_subplot(gs[1, 1]),
            'disk_io': fig.add_subplot(gs[1, 2]),
            'network_io': fig.add_subplot(gs[2, 0]),
            'gpu_usage': fig.add_subplot(gs[2, 1]),
            'temperatures': fig.add_subplot(gs[2, 2]),
            'process_info': fig.add_subplot(gs[3, 0]),
            'system_info': fig.add_subplot(gs[3, 1]),
            'resource_trends': fig.add_subplot(gs[3, 2])
        }
        
        self.figures['hardware'] = fig
        self._add_canvas_to_frame(fig, frame, 'hardware')
    
    def _create_pathos_tab(self):
        """Create real-time Pathos state tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="💝 Pathos State")
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.suptitle('Pathos Affective State Dynamics', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.pathos_axes = {
            'state_norm': fig.add_subplot(2, 2, 1),
            'reward': fig.add_subplot(2, 2, 2),
            'heatmap': fig.add_subplot(2, 2, 3),
            'phase_space': fig.add_subplot(2, 2, 4)
        }
        
        # Configure axes
        self.pathos_axes['state_norm'].set_title('State Vector Norm Over Time')
        self.pathos_axes['state_norm'].set_xlabel('Time Steps')
        self.pathos_axes['state_norm'].set_ylabel('||F(t)||')
        
        self.pathos_axes['reward'].set_title('Internal Reward Over Time')
        self.pathos_axes['reward'].set_xlabel('Time Steps')
        self.pathos_axes['reward'].set_ylabel('Internal Reward')
        
        self.pathos_axes['heatmap'].set_title('State Components (Recent)')
        self.pathos_axes['phase_space'].set_title('Phase Space (Norm vs Reward)')
        
        self.figures['pathos'] = fig
        self._add_canvas_to_frame(fig, frame, 'pathos')
    
    def _create_memory_tab(self):
        """Create memory network visualization tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🧠 Memory Network")
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.suptitle('Memory Network Structure and Relationships', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.memory_axes = {
            'network': fig.add_subplot(1, 2, 1),
            'statistics': fig.add_subplot(1, 2, 2)
        }
        
        self.memory_axes['network'].set_title('Memory Network Graph')
        self.memory_axes['statistics'].set_title('Memory Statistics')
        
        self.figures['memory'] = fig
        self._add_canvas_to_frame(fig, frame, 'memory')
    
    def _create_preference_tab(self):
        """Create preference drift analysis tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="📊 Preference Drift")
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.suptitle('Preference Drift Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.preference_axes = {
            'weights': fig.add_subplot(2, 2, 1),
            'drift_dist': fig.add_subplot(2, 2, 2),
            'volatility': fig.add_subplot(2, 2, 3),
            'stability': fig.add_subplot(2, 2, 4)
        }
        
        # Configure axes
        self.preference_axes['weights'].set_title('Preference Weights Over Time')
        self.preference_axes['drift_dist'].set_title('Drift Magnitude Distribution')
        self.preference_axes['volatility'].set_title('Volatility by Category')
        self.preference_axes['stability'].set_title('Stability Scores')
        
        self.figures['preference'] = fig
        self._add_canvas_to_frame(fig, frame, 'preference')
    
    def _create_attractor_tab(self):
        """Create attractor pattern analysis tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🌀 Attractor Patterns")
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.suptitle('Attractor Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.attractor_axes = {
            'count': fig.add_subplot(2, 2, 1),
            'strength': fig.add_subplot(2, 2, 2),
            'recognition': fig.add_subplot(2, 2, 3),
            'dynamics': fig.add_subplot(2, 2, 4)
        }
        
        # Configure axes
        self.attractor_axes['count'].set_title('Attractor Count Over Time')
        self.attractor_axes['strength'].set_title('Strongest Attractor Strength')
        self.attractor_axes['recognition'].set_title('Pattern Recognition Events')
        self.attractor_axes['dynamics'].set_title('Attractor Dynamics Phase Space')
        
        self.figures['attractor'] = fig
        self._add_canvas_to_frame(fig, frame, 'attractor')
    
    def _create_performance_tab(self):
        """Create performance metrics tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="⚡ Performance")
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.suptitle('Performance Metrics and Trends', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.performance_axes = {
            'cycle_time': fig.add_subplot(2, 3, 1),
            'memory_usage': fig.add_subplot(2, 3, 2),
            'tool_success': fig.add_subplot(2, 3, 3),
            'throughput': fig.add_subplot(2, 3, 4),
            'efficiency': fig.add_subplot(2, 3, 5),
            'trends': fig.add_subplot(2, 3, 6)
        }
        
        # Configure axes
        self.performance_axes['cycle_time'].set_title('Cycle Time')
        self.performance_axes['memory_usage'].set_title('Memory Usage')
        self.performance_axes['tool_success'].set_title('Tool Success Rate')
        self.performance_axes['throughput'].set_title('Throughput')
        self.performance_axes['efficiency'].set_title('Efficiency Score')
        self.performance_axes['trends'].set_title('Performance Trends')
        
        self.figures['performance'] = fig
        self._add_canvas_to_frame(fig, frame, 'performance')
    
    def _create_thread_management_tab(self):
        """Create thread management and monitoring tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="🧵 Thread Management")
        
        # Create matplotlib figure for thread visualizations
        fig = Figure(figsize=(14, 10), facecolor='white')
        
        # Create subplots for different thread metrics
        self.thread_axes = {
            'thread_count': fig.add_subplot(4, 3, 1),
            'thread_types': fig.add_subplot(4, 3, 2),
            'thread_states': fig.add_subplot(4, 3, 3),
            'cpu_usage': fig.add_subplot(4, 3, 4),
            'memory_usage': fig.add_subplot(4, 3, 5),
            'thread_lifetime': fig.add_subplot(4, 3, 6),
            'performance_trends': fig.add_subplot(4, 3, 7),
            'resource_limits': fig.add_subplot(4, 3, 8),
            'thread_timeline': fig.add_subplot(4, 3, 9),
            'thread_pools': fig.add_subplot(4, 3, 10),
            'pool_utilization': fig.add_subplot(4, 3, 11),
            'pool_activity': fig.add_subplot(4, 3, 12)
        }
        
        # Set titles
        self.thread_axes['thread_count'].set_title('Active Thread Count')
        self.thread_axes['thread_types'].set_title('Threads by Type')
        self.thread_axes['thread_states'].set_title('Thread States')
        self.thread_axes['cpu_usage'].set_title('Thread CPU Usage')
        self.thread_axes['memory_usage'].set_title('Thread Memory Usage')
        self.thread_axes['thread_lifetime'].set_title('Thread Lifetimes')
        self.thread_axes['performance_trends'].set_title('Thread Performance Trends')
        self.thread_axes['resource_limits'].set_title('Resource Limit Status')
        self.thread_axes['thread_timeline'].set_title('Thread Creation Timeline')
        self.thread_axes['thread_pools'].set_title('Thread Pools Overview')
        self.thread_axes['pool_utilization'].set_title('Pool Worker Utilization')
        self.thread_axes['pool_activity'].set_title('Pool Activity Trends')
        
        self.figures['threads'] = fig
        self._add_canvas_to_frame(fig, frame, 'threads')
        
        # Add thread control panel
        control_frame = ttk.LabelFrame(frame, text="Thread Controls", padding="5")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Thread statistics display
        stats_frame = ttk.Frame(control_frame)
        stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.thread_stats_var = tk.StringVar()
        self.thread_stats_var.set("Thread Manager: Initializing...")
        stats_label = ttk.Label(stats_frame, textvariable=self.thread_stats_var, font=('Consolas', 9))
        stats_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="🔄 Refresh", 
                  command=self._refresh_thread_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📊 Export Thread Data", 
                  command=self._export_thread_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⚠️ Emergency Stop All", 
                  command=self._emergency_stop_threads).pack(side=tk.LEFT, padx=2)
        
        # Initialize thread manager integration
        self._initialize_thread_manager()
    
    def _create_log_audit_tab(self):
        """Create comprehensive log and audit trail tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="📋 Log & Audit")
        
        # Create main container with splitter
        main_container = ttk.PanedWindow(frame, orient=tk.VERTICAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top section: Log viewer
        log_frame = ttk.LabelFrame(main_container, text="System Log Stream", padding="5")
        main_container.add(log_frame, weight=3)
        
        # Log text widget with scrollbar
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for logs
        self.log_text = tk.Text(log_container, wrap=tk.WORD, font=('Consolas', 9),
                               bg='black', fg='lightgreen', insertbackground='white')
        
        # Scrollbar for log text
        log_scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Pack log components
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom section: Event summary and controls
        summary_frame = ttk.LabelFrame(main_container, text="Event Summary & Controls", padding="5")
        main_container.add(summary_frame, weight=1)
        
        # Create notebook for different log views
        log_notebook = ttk.Notebook(summary_frame)
        log_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Event Statistics Tab
        stats_frame = ttk.Frame(log_notebook, padding="5")
        log_notebook.add(stats_frame, text="📊 Statistics")
        
        # Create statistics display
        self._create_log_statistics_display(stats_frame)
        
        # Event Filter Tab
        filter_frame = ttk.Frame(log_notebook, padding="5")
        log_notebook.add(filter_frame, text="🔍 Filters")
        
        # Create filter controls
        self._create_log_filter_controls(filter_frame)
        
        # Export & Controls Tab
        export_frame = ttk.Frame(log_notebook, padding="5")
        log_notebook.add(export_frame, text="💾 Export")
        
        # Create export controls
        self._create_log_export_controls(export_frame)
        
        # Initialize log system
        self._initialize_log_system()
    
    def _create_log_statistics_display(self, parent):
        """Create log statistics display"""
        # Statistics grid
        stats_grid = ttk.Frame(parent)
        stats_grid.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid
        for i in range(3):
            stats_grid.columnconfigure(i, weight=1)
        
        # Event counters
        self.log_stats = {
            'total_events': tk.StringVar(value="Total Events: 0"),
            'session_events': tk.StringVar(value="Session Events: 0"),
            'tool_events': tk.StringVar(value="Tool Executions: 0"),
            'error_events': tk.StringVar(value="Errors: 0"),
            'thread_events': tk.StringVar(value="Thread Events: 0"),
            'hardware_events': tk.StringVar(value="Hardware Events: 0"),
            'memory_events': tk.StringVar(value="Memory Events: 0"),
            'pathos_events': tk.StringVar(value="Pathos Events: 0"),
            'ethos_events': tk.StringVar(value="Ethos Events: 0")
        }
        
        # Create stat labels in grid
        row = 0
        col = 0
        for stat_name, stat_var in self.log_stats.items():
            label = ttk.Label(stats_grid, textvariable=stat_var, font=('Arial', 10))
            label.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        # Real-time event rate
        rate_frame = ttk.Frame(parent)
        rate_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.event_rate_var = tk.StringVar(value="Event Rate: 0 events/min")
        rate_label = ttk.Label(rate_frame, textvariable=self.event_rate_var, 
                              font=('Arial', 11, 'bold'), foreground='blue')
        rate_label.pack(side=tk.LEFT)
        
        # Last event timestamp
        self.last_event_var = tk.StringVar(value="Last Event: Never")
        last_label = ttk.Label(rate_frame, textvariable=self.last_event_var, font=('Arial', 10))
        last_label.pack(side=tk.RIGHT)
    
    def _create_log_filter_controls(self, parent):
        """Create log filtering controls"""
        # Log level filter
        level_frame = ttk.LabelFrame(parent, text="Log Level Filter", padding="5")
        level_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.log_level_vars = {}
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        colors = ['gray', 'blue', 'orange', 'red', 'darkred']
        
        for i, (level, color) in enumerate(zip(levels, colors)):
            var = tk.BooleanVar(value=True if level in ['INFO', 'WARNING', 'ERROR'] else False)
            self.log_level_vars[level] = var
            
            cb = ttk.Checkbutton(level_frame, text=level, variable=var,
                               command=self._update_log_filter)
            cb.grid(row=0, column=i, padx=5, sticky=tk.W)
        
        # Component filter
        component_frame = ttk.LabelFrame(parent, text="Component Filter", padding="5")
        component_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.component_vars = {}
        components = ['Session', 'Agent', 'Tools', 'Memory', 'Pathos', 'Ethos', 'Hardware', 'Threads']
        
        for i, component in enumerate(components):
            var = tk.BooleanVar(value=True)
            self.component_vars[component] = var
            
            cb = ttk.Checkbutton(component_frame, text=component, variable=var,
                               command=self._update_log_filter)
            cb.grid(row=i//4, column=i%4, padx=5, sticky=tk.W)
        
        # Search filter
        search_frame = ttk.LabelFrame(parent, text="Search Filter", padding="5")
        search_frame.pack(fill=tk.X)
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(0, 5))
        search_entry.bind('<KeyRelease>', self._on_search_change)
        
        ttk.Button(search_frame, text="Clear", command=self._clear_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)
    
    def _create_log_export_controls(self, parent):
        """Create log export and control options"""
        # Auto-scroll control
        scroll_frame = ttk.LabelFrame(parent, text="Display Options", padding="5")
        scroll_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_frame, text="Auto-scroll to latest", 
                       variable=self.auto_scroll_var).pack(side=tk.LEFT)
        
        self.show_timestamps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_frame, text="Show timestamps", 
                       variable=self.show_timestamps_var).pack(side=tk.LEFT, padx=(20, 0))
        
        self.color_coding_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scroll_frame, text="Color coding", 
                       variable=self.color_coding_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Buffer size control
        buffer_frame = ttk.Frame(scroll_frame)
        buffer_frame.pack(side=tk.RIGHT)
        
        ttk.Label(buffer_frame, text="Buffer size:").pack(side=tk.LEFT)
        self.buffer_size_var = tk.IntVar(value=1000)
        buffer_spinbox = ttk.Spinbox(buffer_frame, from_=100, to=10000, increment=100,
                                   width=8, textvariable=self.buffer_size_var)
        buffer_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Export controls
        export_frame = ttk.LabelFrame(parent, text="Export Options", padding="5")
        export_frame.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="📄 Export Current Log", 
                  command=self._export_current_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="📊 Export Statistics", 
                  command=self._export_log_statistics).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="🔍 Export Filtered Log", 
                  command=self._export_filtered_log).pack(side=tk.LEFT, padx=(0, 5))
        
        # Session log controls
        session_frame = ttk.Frame(export_frame)
        session_frame.pack(side=tk.RIGHT)
        
        ttk.Button(session_frame, text="📋 Copy to Clipboard", 
                  command=self._copy_log_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(session_frame, text="🔄 Refresh", 
                  command=self._refresh_log_display).pack(side=tk.LEFT)
    
    def _initialize_log_system(self):
        """Initialize the logging and audit system"""
        # Log buffer for storing events
        self.log_buffer = deque(maxlen=1000)
        self.event_counts = {
            'total': 0, 'session': 0, 'tool': 0, 'error': 0,
            'thread': 0, 'hardware': 0, 'memory': 0, 'pathos': 0, 'ethos': 0
        }
        self.last_event_time = None
        self.event_rate_history = deque(maxlen=60)  # Last 60 seconds
        
        # Configure text widget tags for color coding
        self.log_text.tag_configure('DEBUG', foreground='gray')
        self.log_text.tag_configure('INFO', foreground='lightblue')
        self.log_text.tag_configure('WARNING', foreground='yellow')
        self.log_text.tag_configure('ERROR', foreground='red')
        self.log_text.tag_configure('CRITICAL', foreground='magenta')
        self.log_text.tag_configure('timestamp', foreground='cyan')
        self.log_text.tag_configure('component', foreground='lightgreen')
        
        # Set up log capture from various system components
        self._setup_log_capture()
        
        # Start log update thread
        self.log_update_thread = threading.Thread(target=self._log_update_loop, daemon=True)
        self.log_update_thread.name = "LogAuditUpdater"
        
        try:
            from .thread_manager import register_thread
            register_thread(self.log_update_thread, "dashboard", "LogAuditTab")
        except ImportError:
            pass
        
        self.log_update_thread.start()
        
        print("✅ Log & Audit system initialized")
    
    def _setup_log_capture(self):
        """Setup log capture from system components"""
        # Capture from session manager
        if hasattr(self.session_manager, 'add_state_change_callback'):
            self.session_manager.add_state_change_callback(self._on_session_event)
            self.session_manager.add_cycle_complete_callback(self._on_cycle_event)
            self.session_manager.add_error_callback(self._on_error_event)
        
        # Capture from thread manager
        if hasattr(self, 'thread_manager') and self.thread_manager:
            self.thread_manager.add_performance_callback(self._on_thread_event)
        
        # Capture from hardware monitor
        if hasattr(self.hardware_monitor, 'add_callback'):
            # Add callback if hardware monitor supports it
            pass
        
        # Add system-wide log capture
        self._add_log_entry("SYSTEM", "INFO", "Log & Audit system started")
    
    def _log_update_loop(self):
        """Main loop for updating log display"""
        while self.is_monitoring:
            try:
                # Update event rate calculation
                current_time = time.time()
                self.event_rate_history.append(current_time)
                
                # Calculate events per minute
                if len(self.event_rate_history) > 1:
                    time_span = self.event_rate_history[-1] - self.event_rate_history[0]
                    if time_span > 0:
                        events_per_minute = (len(self.event_rate_history) / time_span) * 60
                        self.root.after(0, lambda: self.event_rate_var.set(f"Event Rate: {events_per_minute:.1f} events/min"))
                
                # Update statistics
                self.root.after(0, self._update_log_statistics)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Log update error: {e}")
                time.sleep(1)
    
    def _add_log_entry(self, component: str, level: str, message: str, details: str = ""):
        """Add a new log entry to the system"""
        timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp,
            'component': component,
            'level': level,
            'message': message,
            'details': details
        }
        
        # Add to buffer
        self.log_buffer.append(log_entry)
        
        # Update counters
        self.event_counts['total'] += 1
        component_key = component.lower()
        if component_key in self.event_counts:
            self.event_counts[component_key] += 1
        
        if level in ['ERROR', 'CRITICAL']:
            self.event_counts['error'] += 1
        
        self.last_event_time = timestamp
        
        # Schedule GUI update
        self.root.after(0, lambda: self._display_log_entry(log_entry))
    
    def _display_log_entry(self, entry):
        """Display a log entry in the text widget"""
        if not hasattr(self, 'log_text'):
            return
        
        # Check filters
        if not self._passes_filters(entry):
            return
        
        # Format entry
        timestamp_str = entry['timestamp'].strftime("%H:%M:%S.%f")[:-3]
        component_str = f"[{entry['component']}]"
        level_str = f"{entry['level']}"
        message_str = entry['message']
        
        # Build display line
        if self.show_timestamps_var.get():
            line = f"{timestamp_str} "
        else:
            line = ""
        
        line += f"{component_str:12} {level_str:8} | {message_str}"
        
        if entry['details']:
            line += f" | {entry['details']}"
        
        line += "\n"
        
        # Insert with color coding
        if self.color_coding_var.get():
            # Insert timestamp
            if self.show_timestamps_var.get():
                self.log_text.insert(tk.END, f"{timestamp_str} ", 'timestamp')
            
            # Insert component
            self.log_text.insert(tk.END, f"{component_str:12} ", 'component')
            
            # Insert level with color
            self.log_text.insert(tk.END, f"{level_str:8} | ", level_str)
            
            # Insert message
            self.log_text.insert(tk.END, f"{message_str}")
            
            if entry['details']:
                self.log_text.insert(tk.END, f" | {entry['details']}")
            
            self.log_text.insert(tk.END, "\n")
        else:
            self.log_text.insert(tk.END, line)
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)
        
        # Limit buffer size
        buffer_size = self.buffer_size_var.get()
        lines = self.log_text.get("1.0", tk.END).count('\n')
        if lines > buffer_size:
            # Remove old lines
            excess_lines = lines - buffer_size
            self.log_text.delete("1.0", f"{excess_lines + 1}.0")
    
    def _passes_filters(self, entry):
        """Check if log entry passes current filters"""
        # Level filter
        if entry['level'] not in self.log_level_vars or not self.log_level_vars[entry['level']].get():
            return False
        
        # Component filter
        component_match = False
        for comp_name, var in self.component_vars.items():
            if var.get() and comp_name.lower() in entry['component'].lower():
                component_match = True
                break
        
        if not component_match:
            return False
        
        # Search filter
        search_text = self.search_var.get().lower()
        if search_text:
            searchable_text = f"{entry['message']} {entry['details']}".lower()
            if search_text not in searchable_text:
                return False
        
        return True
    
    def _update_log_statistics(self):
        """Update log statistics display"""
        try:
            self.log_stats['total_events'].set(f"Total Events: {self.event_counts['total']}")
            self.log_stats['session_events'].set(f"Session Events: {self.event_counts['session']}")
            self.log_stats['tool_events'].set(f"Tool Executions: {self.event_counts['tool']}")
            self.log_stats['error_events'].set(f"Errors: {self.event_counts['error']}")
            self.log_stats['thread_events'].set(f"Thread Events: {self.event_counts['thread']}")
            self.log_stats['hardware_events'].set(f"Hardware Events: {self.event_counts['hardware']}")
            self.log_stats['memory_events'].set(f"Memory Events: {self.event_counts['memory']}")
            self.log_stats['pathos_events'].set(f"Pathos Events: {self.event_counts['pathos']}")
            self.log_stats['ethos_events'].set(f"Ethos Events: {self.event_counts['ethos']}")
            
            if self.last_event_time:
                self.last_event_var.set(f"Last Event: {self.last_event_time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Statistics update error: {e}")
    
    # Event handlers for different system components
    def _on_session_event(self, state):
        """Handle session state change events"""
        self._add_log_entry("SESSION", "INFO", f"State changed to: {state.value}")
    
    def _on_cycle_event(self, cycle, data):
        """Handle agent cycle completion events"""
        duration = data.get('duration', 0)
        self._add_log_entry("AGENT", "INFO", f"Cycle {cycle} completed", f"Duration: {duration:.3f}s")
    
    def _on_error_event(self, error):
        """Handle error events"""
        self._add_log_entry("SYSTEM", "ERROR", f"Error occurred: {str(error)}")
    
    def _on_thread_event(self, metrics):
        """Handle thread performance events"""
        active_threads = metrics.get('active_threads', 0)
        self._add_log_entry("THREADS", "DEBUG", f"Performance update", f"Active: {active_threads}")
    
    # Filter and control methods
    def _update_log_filter(self):
        """Update log display based on current filters"""
        # Clear current display
        self.log_text.delete("1.0", tk.END)
        
        # Redisplay filtered entries
        for entry in list(self.log_buffer):
            if self._passes_filters(entry):
                self._display_log_entry(entry)
    
    def _on_search_change(self, event=None):
        """Handle search text changes"""
        # Debounce search updates
        if hasattr(self, '_search_timer'):
            self.root.after_cancel(self._search_timer)
        
        self._search_timer = self.root.after(500, self._update_log_filter)
    
    def _clear_search(self):
        """Clear search filter"""
        self.search_var.set("")
        self._update_log_filter()
    
    def _clear_log(self):
        """Clear all log entries"""
        if messagebox.askyesno("Clear Log", "Are you sure you want to clear all log entries?"):
            self.log_text.delete("1.0", tk.END)
            self.log_buffer.clear()
            self.event_counts = {key: 0 for key in self.event_counts}
            self._update_log_statistics()
            self._add_log_entry("SYSTEM", "INFO", "Log cleared by user")
    
    # Export methods
    def _export_current_log(self):
        """Export current log to file"""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Current Log"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get("1.0", tk.END))
                
                self._add_log_entry("SYSTEM", "INFO", f"Log exported to: {filename}")
                
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Export failed: {str(e)}")
    
    def _export_log_statistics(self):
        """Export log statistics to JSON"""
        try:
            from tkinter import filedialog
            import json
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Log Statistics"
            )
            
            if filename:
                stats_data = {
                    'export_time': datetime.now().isoformat(),
                    'event_counts': self.event_counts.copy(),
                    'total_entries': len(self.log_buffer),
                    'buffer_size': self.buffer_size_var.get(),
                    'last_event': self.last_event_time.isoformat() if self.last_event_time else None
                }
                
                with open(filename, 'w') as f:
                    json.dump(stats_data, f, indent=2)
                
                self._add_log_entry("SYSTEM", "INFO", f"Statistics exported to: {filename}")
                
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Statistics export failed: {str(e)}")
    
    def _export_filtered_log(self):
        """Export only filtered log entries"""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Filtered Log"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    for entry in self.log_buffer:
                        if self._passes_filters(entry):
                            timestamp_str = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            line = f"{timestamp_str} [{entry['component']}] {entry['level']} | {entry['message']}"
                            if entry['details']:
                                line += f" | {entry['details']}"
                            f.write(line + "\n")
                
                self._add_log_entry("SYSTEM", "INFO", f"Filtered log exported to: {filename}")
                
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Filtered export failed: {str(e)}")
    
    def _copy_log_to_clipboard(self):
        """Copy current log display to clipboard"""
        try:
            log_content = self.log_text.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self._add_log_entry("SYSTEM", "INFO", "Log copied to clipboard")
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Clipboard copy failed: {str(e)}")
    
    def _refresh_log_display(self):
        """Refresh the log display"""
        self._update_log_filter()
        self._add_log_entry("SYSTEM", "INFO", "Log display refreshed")
    
    def _add_canvas_to_frame(self, fig: Figure, frame: ttk.Frame, name: str):
        """Add matplotlib canvas to tkinter frame"""
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        
        # Create toolbar
        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Pack canvas
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Store canvas
        self.canvases[name] = canvas
    
    def _open_agent_config(self):
        """Open agent configuration dialog"""
        config_window = tk.Toplevel(self.root)
        config_window.title("Agent Configuration")
        config_window.geometry("500x600")
        config_window.transient(self.root)
        config_window.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(config_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Agent Identity
        ttk.Label(main_frame, text="Agent Identity:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        identity_var = tk.StringVar(value=self.session_config.agent_identity)
        identity_entry = ttk.Entry(main_frame, textvariable=identity_var, width=50)
        identity_entry.pack(fill=tk.X, pady=(0, 15))
        
        # Pathos Dimension
        ttk.Label(main_frame, text="Pathos State Dimension:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        pathos_dim_var = tk.IntVar(value=self.session_config.pathos_dimension)
        pathos_dim_spinbox = ttk.Spinbox(main_frame, from_=32, to=512, increment=32, 
                                        textvariable=pathos_dim_var, width=20)
        pathos_dim_spinbox.pack(anchor=tk.W, pady=(0, 15))
        
        # Memory Capacity
        ttk.Label(main_frame, text="Memory Capacity:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        memory_cap_var = tk.IntVar(value=self.session_config.memory_capacity)
        memory_cap_spinbox = ttk.Spinbox(main_frame, from_=100, to=10000, increment=100,
                                        textvariable=memory_cap_var, width=20)
        memory_cap_spinbox.pack(anchor=tk.W, pady=(0, 15))
        
        # Exploration Rate
        ttk.Label(main_frame, text="Exploration Rate:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        exploration_var = tk.DoubleVar(value=self.session_config.exploration_rate)
        exploration_scale = ttk.Scale(main_frame, from_=0.0, to=1.0, variable=exploration_var,
                                     orient=tk.HORIZONTAL, length=300)
        exploration_scale.pack(anchor=tk.W, pady=(0, 5))
        exploration_label = ttk.Label(main_frame, text="")
        exploration_label.pack(anchor=tk.W, pady=(0, 15))
        
        def update_exploration_label(*args):
            exploration_label.config(text=f"Current: {exploration_var.get():.2f}")
        exploration_var.trace('w', update_exploration_label)
        update_exploration_label()
        
        # Safety Settings
        ttk.Label(main_frame, text="Safety Settings:", style='Header.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        # Max Memory Usage
        ttk.Label(main_frame, text="Max Memory Usage (MB):").pack(anchor=tk.W, pady=(0, 2))
        max_memory_var = tk.IntVar(value=self.session_config.max_memory_usage_mb)
        max_memory_spinbox = ttk.Spinbox(main_frame, from_=256, to=8192, increment=256,
                                        textvariable=max_memory_var, width=20)
        max_memory_spinbox.pack(anchor=tk.W, pady=(0, 10))
        
        # Max CPU Percent
        ttk.Label(main_frame, text="Max CPU Usage (%):").pack(anchor=tk.W, pady=(0, 2))
        max_cpu_var = tk.DoubleVar(value=self.session_config.max_cpu_percent)
        max_cpu_spinbox = ttk.Spinbox(main_frame, from_=10.0, to=100.0, increment=5.0,
                                     textvariable=max_cpu_var, width=20)
        max_cpu_spinbox.pack(anchor=tk.W, pady=(0, 10))
        
        # Checkboxes
        hardware_monitoring_var = tk.BooleanVar(value=self.session_config.enable_hardware_monitoring)
        ttk.Checkbutton(main_frame, text="Enable Hardware Monitoring", 
                       variable=hardware_monitoring_var).pack(anchor=tk.W, pady=2)
        
        save_data_var = tk.BooleanVar(value=self.session_config.save_session_data)
        ttk.Checkbutton(main_frame, text="Save Session Data", 
                       variable=save_data_var).pack(anchor=tk.W, pady=2)
        
        emergency_stop_var = tk.BooleanVar(value=self.session_config.emergency_stop_on_error)
        ttk.Checkbutton(main_frame, text="Emergency Stop on Error", 
                       variable=emergency_stop_var).pack(anchor=tk.W, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def save_config():
            self.session_config.agent_identity = identity_var.get()
            self.session_config.pathos_dimension = pathos_dim_var.get()
            self.session_config.memory_capacity = memory_cap_var.get()
            self.session_config.exploration_rate = exploration_var.get()
            self.session_config.max_memory_usage_mb = max_memory_var.get()
            self.session_config.max_cpu_percent = max_cpu_var.get()
            self.session_config.enable_hardware_monitoring = hardware_monitoring_var.get()
            self.session_config.save_session_data = save_data_var.get()
            self.session_config.emergency_stop_on_error = emergency_stop_var.get()
            config_window.destroy()
            messagebox.showinfo("Configuration", "Agent configuration saved successfully!")
        
        ttk.Button(button_frame, text="Save", command=save_config).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=config_window.destroy).pack(side=tk.RIGHT)
    
    def _create_status_bar(self, parent):
        """Create comprehensive status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(1, weight=1)
        
        # Left side - Status message
        self.status_var = tk.StringVar(value="Ready - Configure and start a session")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                style='Status.TLabel')
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Center - Session progress
        self.progress_var = tk.StringVar(value="")
        progress_label = ttk.Label(status_frame, textvariable=self.progress_var,
                                  style='Status.TLabel')
        progress_label.grid(row=0, column=1, padx=(20, 20))
        
        # Right side - Connection indicators
        indicators_frame = ttk.Frame(status_frame)
        indicators_frame.grid(row=0, column=2, sticky=tk.E)
        
        # Session indicator
        self.session_indicator_var = tk.StringVar(value="🔴 No Session")
        session_indicator = ttk.Label(indicators_frame, textvariable=self.session_indicator_var,
                                     style='Status.TLabel')
        session_indicator.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Monitoring indicator
        self.monitoring_indicator_var = tk.StringVar(value="📊 Monitoring Off")
        monitoring_indicator = ttk.Label(indicators_frame, textvariable=self.monitoring_indicator_var,
                                        style='Status.TLabel')
        monitoring_indicator.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Hardware indicator
        self.hardware_indicator_var = tk.StringVar(value="🖥️ Hardware Off")
        hardware_indicator = ttk.Label(indicators_frame, textvariable=self.hardware_indicator_var,
                                      style='Status.TLabel')
        hardware_indicator.pack(side=tk.RIGHT)
    
    def _toggle_session(self):
        """Toggle session start/stop"""
        if self.session_manager.state == SessionState.IDLE:
            self._start_session()
        elif self.session_manager.state in [SessionState.RUNNING, SessionState.PAUSED]:
            self._stop_session()
    
    def _start_session(self):
        """Start agent session"""
        try:
            # Update configuration from GUI
            self.session_config.duration_minutes = self.duration_var.get()
            max_cycles_str = self.max_cycles_var.get().strip()
            self.session_config.max_cycles = int(max_cycles_str) if max_cycles_str else None
            self.session_config.cycle_delay = self.cycle_delay_var.get()
            
            # Configure session manager
            self.session_manager.configure_session(self.session_config)
            
            # Start session
            if self.session_manager.start_session():
                self.session_button.config(text="⏹️ Stop Session")
                self.pause_button.config(state='normal')
                self.status_var.set("Session starting...")
                
                # Update collector reference
                self.collector = self.session_manager.collector
                
                # Start monitoring if not already running
                if not self.is_monitoring:
                    self._start_monitoring()
            else:
                messagebox.showerror("Session Error", "Failed to start session")
                
        except ValueError as e:
            messagebox.showerror("Configuration Error", f"Invalid configuration: {e}")
        except Exception as e:
            messagebox.showerror("Session Error", f"Failed to start session: {e}")
    
    def _stop_session(self):
        """Stop agent session"""
        self.session_manager.stop_session()
        self.session_button.config(text="▶️ Start Session")
        self.pause_button.config(state='disabled', text="⏸️ Pause")
        self.status_var.set("Session stopped")
    
    def _toggle_pause(self):
        """Toggle session pause/resume"""
        if self.session_manager.state == SessionState.RUNNING:
            self.session_manager.pause_session()
            self.pause_button.config(text="▶️ Resume")
        elif self.session_manager.state == SessionState.PAUSED:
            self.session_manager.resume_session()
            self.pause_button.config(text="⏸️ Pause")
    
    def _emergency_stop(self):
        """Emergency stop session"""
        if messagebox.askyesno("Emergency Stop", "Are you sure you want to emergency stop the session?"):
            self.session_manager.emergency_stop("User requested emergency stop")
            self.session_button.config(text="▶️ Start Session")
            self.pause_button.config(state='disabled', text="⏸️ Pause")
            self.status_var.set("Emergency stop executed")
    
    def _on_session_state_change(self, new_state: SessionState):
        """Handle session state changes"""
        state_colors = {
            SessionState.IDLE: ("🔴 No Session", 'Status.TLabel'),
            SessionState.INITIALIZING: ("🟡 Initializing", 'Warning.TLabel'),
            SessionState.RUNNING: ("🟢 Running", 'Success.TLabel'),
            SessionState.PAUSED: ("🟡 Paused", 'Warning.TLabel'),
            SessionState.STOPPING: ("🟡 Stopping", 'Warning.TLabel'),
            SessionState.COMPLETED: ("🔵 Completed", 'Status.TLabel'),
            SessionState.ERROR: ("🔴 Error", 'Error.TLabel')
        }
        
        text, style = state_colors.get(new_state, ("🔴 Unknown", 'Status.TLabel'))
        self.session_indicator_var.set(text)
        
        # Update session info
        if new_state == SessionState.RUNNING:
            stats = self.session_manager.stats
            duration = stats.duration
            if duration:
                minutes = int(duration.total_seconds() // 60)
                seconds = int(duration.total_seconds() % 60)
                self.session_info_var.set(f"Session: {stats.total_cycles} cycles, {minutes}:{seconds:02d}")
        elif new_state == SessionState.IDLE:
            self.session_info_var.set("No active session")
    
    def _on_cycle_complete(self, cycle: int, cycle_data: Dict[str, Any]):
        """Handle cycle completion"""
        stats = self.session_manager.stats
        duration = stats.duration
        if duration:
            minutes = int(duration.total_seconds() // 60)
            seconds = int(duration.total_seconds() % 60)
            rate = stats.cycles_per_minute
            self.progress_var.set(f"Cycle {cycle} | {minutes}:{seconds:02d} | {rate:.1f} cycles/min")
    
    def _on_session_error(self, error: Exception):
        """Handle session errors"""
        messagebox.showerror("Session Error", f"Session error occurred: {error}")
        self.status_var.set(f"Session error: {error}")
    
    def _toggle_monitoring(self):
        """Toggle real-time monitoring"""
        if self.is_monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start real-time monitoring"""
        self.is_monitoring = True
        self.monitor_button.config(text="⏸️ Stop Monitoring")
        self.monitoring_indicator_var.set("📊 Monitoring On")
        
        # Start hardware monitoring
        self.hardware_monitor.start_monitoring(1.0)
        self.hardware_indicator_var.set("🖥️ Hardware On")
        
        # Start update thread with lower priority for better GUI responsiveness
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.name = "DashboardUpdater"
        
        # Register with thread manager
        try:
            from .thread_manager import register_thread
            register_thread(self.update_thread, "dashboard", "InteractiveDashboard")
        except ImportError:
            pass  # Thread manager not available
        
        self.update_thread.start()
        
        # Set lower priority for update thread to keep GUI responsive
        try:
            import os
            if hasattr(os, 'nice'):
                os.nice(1)  # Lower priority on Unix systems
        except:
            pass  # Not available on all systems
    
    def _stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        self.monitor_button.config(text="📊 Start Monitoring")
        self.monitoring_indicator_var.set("📊 Monitoring Off")
        
        # Stop hardware monitoring
        self.hardware_monitor.stop_monitoring()
        self.hardware_indicator_var.set("🖥️ Hardware Off")
    
    def _update_loop(self):
        """Main update loop for real-time monitoring - optimized for GUI responsiveness"""
        update_cycle = 0
        tab_update_order = [
            'health', 'hardware', 'threads', 'performance', 'logs',  # High priority tabs
            'pathos', 'memory', 'preference', 'attractor'            # Lower priority tabs
        ]
        
        while self.is_monitoring:
            try:
                # Only update one tab per cycle to avoid GUI blocking
                current_tab = tab_update_order[update_cycle % len(tab_update_order)]
                
                # Schedule single tab update on main thread
                if current_tab == 'health':
                    self.root.after_idle(self._update_health_tab)
                elif current_tab == 'hardware':
                    self.root.after_idle(self._update_hardware_tab)
                elif current_tab == 'threads':
                    self.root.after_idle(self._update_thread_tab)
                elif current_tab == 'performance':
                    self.root.after_idle(self._update_performance_tab)
                elif current_tab == 'pathos':
                    self.root.after_idle(self._update_pathos_tab)
                elif current_tab == 'memory':
                    self.root.after_idle(self._update_memory_tab)
                elif current_tab == 'preference':
                    self.root.after_idle(self._update_preference_tab)
                elif current_tab == 'attractor':
                    self.root.after_idle(self._update_attractor_tab)
                elif current_tab == 'logs':
                    self.root.after_idle(self._update_log_tab)
                
                update_cycle += 1
                
                # Shorter sleep for more responsive updates
                time.sleep(max(0.5, self.interval_var.get() / len(tab_update_order)))
                
            except Exception as e:
                print(f"Update error: {e}")
                time.sleep(0.5)
    
    def _update_all_visualizations(self):
        """Update all visualization tabs - used for manual refresh only"""
        # For manual updates, stagger them with small delays to avoid blocking
        delay = 0
        for update_func in [
            self._update_health_tab,
            self._update_hardware_tab, 
            self._update_thread_tab,
            self._update_performance_tab,
            self._update_pathos_tab,
            self._update_memory_tab,
            self._update_preference_tab,
            self._update_attractor_tab,
            self._update_log_tab
        ]:
            self.root.after(delay, update_func)
            delay += 100  # 100ms between each update
    
    def _update_health_tab(self):
        """Update enhanced system health tab - optimized for performance"""
        try:
            # Only update if this tab is visible or if forced
            if not hasattr(self, '_last_health_update'):
                self._last_health_update = 0
            
            current_time = time.time()
            if current_time - self._last_health_update < 2.0:  # Limit to every 2 seconds
                return
            
            self._last_health_update = current_time
            
            # Clear only necessary axes to reduce work
            essential_axes = ['overview', 'hardware_summary', 'session_status']
            for ax_name in essential_axes:
                if ax_name in self.health_axes:
                    self.health_axes[ax_name].clear()
            
            # Get current data
            if self.collector:
                metrics_summary = self.collector.get_metrics_summary()
                
                # Only update essential plots for performance
                try:
                    self.health_dashboard._plot_system_overview(
                        self.health_axes['overview'], 
                        metrics_summary
                    )
                except:
                    pass  # Skip if error
            
            # Hardware summary (lightweight)
            self._plot_hardware_summary()
            
            # Session status (lightweight)
            self._plot_session_status()
            
            # Update canvas efficiently
            self.canvases['health'].draw_idle()  # Use draw_idle for better performance
            
        except Exception as e:
            print(f"Health tab update error: {e}")
    
    def _update_hardware_tab(self):
        """Update comprehensive hardware monitoring tab - optimized"""
        try:
            # Rate limiting
            if not hasattr(self, '_last_hardware_update'):
                self._last_hardware_update = 0
            
            current_time = time.time()
            if current_time - self._last_hardware_update < 1.5:  # Limit to every 1.5 seconds
                return
            
            self._last_hardware_update = current_time
            
            # Get hardware data
            hardware_summary = self.hardware_monitor.get_current_summary()
            
            if 'error' not in hardware_summary:
                current_metrics = hardware_summary.get('current_metrics', {})
                
                # Clear and update only essential plots
                essential_plots = ['cpu_usage', 'memory_usage', 'disk_usage']
                for plot_name in essential_plots:
                    if plot_name in self.hardware_axes:
                        self.hardware_axes[plot_name].clear()
                
                # Update essential hardware metrics only
                try:
                    self._plot_cpu_usage(current_metrics.get('cpu', {}))
                    self._plot_memory_usage(current_metrics.get('memory', {}))
                    self._plot_disk_usage(current_metrics.get('disk', {}))
                except Exception as plot_error:
                    print(f"Hardware plot error: {plot_error}")
            
            # Update canvas efficiently
            self.canvases['hardware'].draw_idle()
            
        except Exception as e:
            print(f"Hardware tab update error: {e}")
    
    def _update_pathos_tab(self):
        """Update Pathos state tab"""
        try:
            # Get latest Pathos data from collector
            if hasattr(self.collector, 'pathos_states') and self.collector.pathos_states:
                latest_state = self.collector.pathos_states[-1]
                
                # Add to history
                self.pathos_history.append({
                    'state': latest_state.get('state_vector', np.zeros(10)),
                    'reward': latest_state.get('internal_reward', 0),
                    'timestamp': datetime.now()
                })
                
                # Update plots
                self._plot_pathos_data()
                
                # Update canvas
                self.canvases['pathos'].draw()
                
        except Exception as e:
            print(f"Pathos tab update error: {e}")
    
    def _plot_pathos_data(self):
        """Plot Pathos state data"""
        if len(self.pathos_history) < 2:
            return
        
        # Clear axes
        for ax in self.pathos_axes.values():
            ax.clear()
        
        # Extract data
        states = [entry['state'] for entry in self.pathos_history]
        rewards = [entry['reward'] for entry in self.pathos_history]
        times = list(range(len(states)))
        
        # State norm over time
        state_norms = [np.linalg.norm(state) for state in states]
        self.pathos_axes['state_norm'].plot(times, state_norms, 'b-', linewidth=2)
        self.pathos_axes['state_norm'].set_title('State Vector Norm Over Time')
        self.pathos_axes['state_norm'].grid(True, alpha=0.3)
        
        # Reward over time
        self.pathos_axes['reward'].plot(times, rewards, 'r-', linewidth=2)
        self.pathos_axes['reward'].set_title('Internal Reward Over Time')
        self.pathos_axes['reward'].grid(True, alpha=0.3)
        
        # State heatmap (recent states)
        if len(states) > 1:
            recent_states = np.array(states[-20:])  # Last 20 states
            if recent_states.size > 0:
                im = self.pathos_axes['heatmap'].imshow(
                    recent_states.T, aspect='auto', cmap='coolwarm', interpolation='nearest'
                )
                self.pathos_axes['heatmap'].set_title('State Components (Recent 20)')
                self.pathos_axes['heatmap'].set_xlabel('Time Steps')
                self.pathos_axes['heatmap'].set_ylabel('State Dimensions')
        
        # Phase space plot
        colors = np.arange(len(state_norms))
        scatter = self.pathos_axes['phase_space'].scatter(
            state_norms, rewards, c=colors, cmap='viridis', alpha=0.6
        )
        self.pathos_axes['phase_space'].plot(state_norms, rewards, 'k-', alpha=0.3, linewidth=1)
        self.pathos_axes['phase_space'].set_title('Phase Space (Norm vs Reward)')
        self.pathos_axes['phase_space'].set_xlabel('State Norm')
        self.pathos_axes['phase_space'].set_ylabel('Internal Reward')
    
    def _update_memory_tab(self):
        """Update memory network tab"""
        try:
            if self.memory_traces:
                # Clear axes
                for ax in self.memory_axes.values():
                    ax.clear()
                
                # Plot memory network (simplified for real-time)
                self._plot_memory_network()
                
                # Update canvas
                self.canvases['memory'].draw()
                
        except Exception as e:
            print(f"Memory tab update error: {e}")
    
    def _plot_memory_network(self):
        """Plot simplified memory network"""
        if len(self.memory_traces) < 2:
            self.memory_axes['network'].text(0.5, 0.5, 'Insufficient memory data', 
                                           ha='center', va='center', transform=self.memory_axes['network'].transAxes)
            return
        
        # Simple scatter plot of memory traces
        recent_traces = self.memory_traces[-50:]  # Last 50 traces
        rewards = [trace.reward for trace in recent_traces]
        saliences = [trace.salience for trace in recent_traces]
        
        scatter = self.memory_axes['network'].scatter(rewards, saliences, 
                                                    alpha=0.6, s=60, c=range(len(rewards)), cmap='viridis')
        self.memory_axes['network'].set_xlabel('Reward')
        self.memory_axes['network'].set_ylabel('Salience')
        self.memory_axes['network'].set_title('Memory Traces (Recent 50)')
        
        # Memory statistics
        if len(recent_traces) > 0:
            avg_reward = np.mean(rewards)
            avg_salience = np.mean(saliences)
            
            stats_text = f"""Memory Statistics:
Total Traces: {len(self.memory_traces)}
Recent Traces: {len(recent_traces)}
Avg Reward: {avg_reward:.3f}
Avg Salience: {avg_salience:.3f}"""
            
            self.memory_axes['statistics'].text(0.1, 0.9, stats_text, 
                                              transform=self.memory_axes['statistics'].transAxes,
                                              verticalalignment='top', fontfamily='monospace')
            self.memory_axes['statistics'].axis('off')
    
    def _update_preference_tab(self):
        """Update preference drift tab"""
        try:
            # Use existing preference analysis
            analysis = self.drift_analyzer.analyze_drift_from_collector(self.collector)
            
            if 'error' not in analysis:
                self._plot_preference_analysis(analysis)
                self.canvases['preference'].draw()
                
        except Exception as e:
            print(f"Preference tab update error: {e}")
    
    def _plot_preference_analysis(self, analysis):
        """Plot preference drift analysis"""
        # Clear axes
        for ax in self.preference_axes.values():
            ax.clear()
        
        # Plot preference weights over time
        if self.collector.preference_history:
            for category, history in self.collector.preference_history.items():
                if len(history) > 1:
                    _, weights = zip(*history)
                    self.preference_axes['weights'].plot(range(len(weights)), weights, 
                                                       marker='o', label=category, linewidth=2)
            
            self.preference_axes['weights'].set_title('Preference Weights Over Time')
            self.preference_axes['weights'].set_xlabel('Update Number')
            self.preference_axes['weights'].set_ylabel('Weight Value')
            self.preference_axes['weights'].legend()
            self.preference_axes['weights'].grid(True, alpha=0.3)
        
        # Plot stability scores
        if analysis and 'summary' not in analysis:
            categories = []
            stability_scores = []
            
            for category, cat_analysis in analysis.items():
                if isinstance(cat_analysis, dict) and 'stability_score' in cat_analysis:
                    categories.append(category)
                    stability_scores.append(cat_analysis['stability_score'])
            
            if categories:
                colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' 
                         for score in stability_scores]
                bars = self.preference_axes['stability'].bar(categories, stability_scores, 
                                                           color=colors, alpha=0.7)
                self.preference_axes['stability'].set_title('Category Stability Scores')
                self.preference_axes['stability'].set_ylabel('Stability Score')
                self.preference_axes['stability'].set_ylim(0, 1)
                
                # Rotate labels if needed
                if len(categories) > 3:
                    self.preference_axes['stability'].tick_params(axis='x', rotation=45)
    
    def _update_attractor_tab(self):
        """Update attractor patterns tab"""
        try:
            # Use existing attractor analysis
            analysis = self.attractor_detector.detect_patterns_from_collector(self.collector)
            
            if 'error' not in analysis:
                self._plot_attractor_analysis()
                self.canvases['attractor'].draw()
                
        except Exception as e:
            print(f"Attractor tab update error: {e}")
    
    def _plot_attractor_analysis(self):
        """Plot attractor pattern analysis"""
        if not self.collector.metrics.get('attractor_patterns'):
            for ax in self.attractor_axes.values():
                ax.clear()
                ax.text(0.5, 0.5, 'No attractor data available', 
                       ha='center', va='center', transform=ax.transAxes)
            return
        
        patterns = self.collector.metrics['attractor_patterns']
        cycles = [p['cycle'] for p in patterns]
        attractor_counts = [p['attractor_count'] for p in patterns]
        attractor_strengths = [p['strongest_attractor_strength'] for p in patterns]
        
        # Clear axes
        for ax in self.attractor_axes.values():
            ax.clear()
        
        # Attractor count over time
        self.attractor_axes['count'].plot(cycles, attractor_counts, 'b-o', linewidth=2, markersize=4)
        self.attractor_axes['count'].set_title('Attractor Count Over Time')
        self.attractor_axes['count'].set_xlabel('Cycle')
        self.attractor_axes['count'].set_ylabel('Number of Attractors')
        self.attractor_axes['count'].grid(True, alpha=0.3)
        
        # Attractor strength over time
        self.attractor_axes['strength'].plot(cycles, attractor_strengths, 'r-o', linewidth=2, markersize=4)
        self.attractor_axes['strength'].set_title('Strongest Attractor Strength')
        self.attractor_axes['strength'].set_xlabel('Cycle')
        self.attractor_axes['strength'].set_ylabel('Attractor Strength')
        self.attractor_axes['strength'].grid(True, alpha=0.3)
    
    def _update_performance_tab(self):
        """Update performance metrics tab"""
        try:
            self._plot_performance_metrics()
            self.canvases['performance'].draw()
        except Exception as e:
            print(f"Performance tab update error: {e}")
    
    def _plot_performance_metrics(self):
        """Plot performance metrics"""
        # Clear axes
        for ax in self.performance_axes.values():
            ax.clear()
        
        # Get cycle timings
        cycle_timings = self.collector.metrics.get('cycle_timings', [])
        
        if cycle_timings:
            cycles = [t['cycle'] for t in cycle_timings]
            durations = [t['total_duration'] for t in cycle_timings]
            
            # Cycle time plot
            self.performance_axes['cycle_time'].plot(cycles, durations, 'b-', linewidth=2)
            self.performance_axes['cycle_time'].set_title('Cycle Time')
            self.performance_axes['cycle_time'].set_xlabel('Cycle')
            self.performance_axes['cycle_time'].set_ylabel('Duration (s)')
            self.performance_axes['cycle_time'].grid(True, alpha=0.3)
            
            # Add trend line
            if len(cycles) > 1:
                z = np.polyfit(cycles, durations, 1)
                p = np.poly1d(z)
                self.performance_axes['cycle_time'].plot(cycles, p(cycles), 'r--', alpha=0.8)
        
        # Tool success rate
        tool_usage = self.collector.metrics.get('tool_usage', [])
        if tool_usage:
            from collections import defaultdict
            tool_stats = defaultdict(lambda: {'success': 0, 'total': 0})
            
            for usage in tool_usage:
                tool_name = usage['tool_name']
                tool_stats[tool_name]['total'] += 1
                if usage['success']:
                    tool_stats[tool_name]['success'] += 1
            
            if tool_stats:
                tools = list(tool_stats.keys())[:10]  # Top 10 tools
                success_rates = [tool_stats[tool]['success'] / tool_stats[tool]['total'] 
                               for tool in tools]
                
                colors = ['green' if rate > 0.7 else 'orange' if rate > 0.4 else 'red' 
                         for rate in success_rates]
                
                bars = self.performance_axes['tool_success'].bar(range(len(tools)), success_rates, 
                                                               color=colors, alpha=0.7)
                self.performance_axes['tool_success'].set_title('Tool Success Rate')
                self.performance_axes['tool_success'].set_ylabel('Success Rate')
                self.performance_axes['tool_success'].set_xticks(range(len(tools)))
                self.performance_axes['tool_success'].set_xticklabels(tools, rotation=45, ha='right')
                self.performance_axes['tool_success'].set_ylim(0, 1)
    
    def _update_log_tab(self):
        """Update log and audit tab - lightweight update"""
        try:
            # Rate limiting for log tab
            if not hasattr(self, '_last_log_update'):
                self._last_log_update = 0
            
            current_time = time.time()
            if current_time - self._last_log_update < 0.5:  # Update every 0.5 seconds max
                return
            
            self._last_log_update = current_time
            
            # Add periodic system status log entries
            if hasattr(self, 'session_manager'):
                session_state = self.session_manager.state
                if session_state.value != getattr(self, '_last_session_state', None):
                    self._add_log_entry("SESSION", "INFO", f"Session state: {session_state.value}")
                    self._last_session_state = session_state.value
            
            # Add hardware status if available
            if hasattr(self, 'hardware_monitor'):
                try:
                    hw_summary = self.hardware_monitor.get_current_summary()
                    if 'error' not in hw_summary:
                        cpu_percent = hw_summary.get('current_metrics', {}).get('cpu', {}).get('cpu_percent', 0)
                        if cpu_percent > 80:  # High CPU usage
                            self._add_log_entry("HARDWARE", "WARNING", f"High CPU usage: {cpu_percent:.1f}%")
                except:
                    pass  # Skip if hardware data not available
            
            # Add thread manager status
            if hasattr(self, 'thread_manager') and self.thread_manager:
                try:
                    summary = self.thread_manager.get_performance_summary()
                    active_threads = summary.get('thread_counts', {}).get('active', 0)
                    if active_threads != getattr(self, '_last_thread_count', 0):
                        self._add_log_entry("THREADS", "DEBUG", f"Active threads: {active_threads}")
                        self._last_thread_count = active_threads
                except:
                    pass  # Skip if thread data not available
            
        except Exception as e:
            print(f"Log tab update error: {e}")
    
    def _manual_update(self):
        """Manually trigger update of all visualizations"""
        self.status_var.set("Updating...")
        self._update_all_visualizations()
        self.status_var.set("Update complete")
    
    def _export_data(self):
        """Export current data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"dashboard_export_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            # Export each figure
            for name, fig in self.figures.items():
                fig.savefig(f"{export_dir}/{name}_dashboard.png", dpi=300, bbox_inches='tight')
            
            # Export data summary
            summary = {
                'timestamp': timestamp,
                'metrics_summary': self.collector.get_metrics_summary(),
                'preference_summary': self.collector.get_preference_drift_summary(),
                'behavior_summary': self.collector.get_behavior_pattern_summary()
            }
            
            with open(f"{export_dir}/data_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            messagebox.showinfo("Export Complete", f"Data exported to {export_dir}")
            self.status_var.set(f"Exported to {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def _plot_hardware_summary(self):
        """Plot hardware summary in health tab"""
        try:
            hardware_summary = self.hardware_monitor.get_current_summary()
            
            if 'error' not in hardware_summary:
                current_metrics = hardware_summary.get('current_metrics', {})
                cpu_data = current_metrics.get('cpu', {})
                memory_data = current_metrics.get('memory', {})
                
                # Simple hardware overview
                cpu_percent = cpu_data.get('cpu_percent', 0)
                memory_percent = memory_data.get('virtual_percent', 0)
                
                categories = ['CPU', 'Memory']
                values = [cpu_percent, memory_percent]
                colors = ['red' if v > 80 else 'orange' if v > 60 else 'green' for v in values]
                
                bars = self.health_axes['hardware_summary'].bar(categories, values, color=colors, alpha=0.7)
                self.health_axes['hardware_summary'].set_title('Hardware Usage')
                self.health_axes['hardware_summary'].set_ylabel('Usage (%)')
                self.health_axes['hardware_summary'].set_ylim(0, 100)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    self.health_axes['hardware_summary'].text(
                        bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{value:.1f}%', ha='center', va='bottom'
                    )
            else:
                self.health_axes['hardware_summary'].text(0.5, 0.5, 'Hardware data unavailable',
                                                         ha='center', va='center', 
                                                         transform=self.health_axes['hardware_summary'].transAxes)
                self.health_axes['hardware_summary'].set_title('Hardware Summary')
        except Exception as e:
            print(f"Hardware summary plot error: {e}")
    
    def _plot_session_status(self):
        """Plot session status in health tab"""
        try:
            session_status = self.session_manager.get_session_status()
            stats = session_status['stats']
            
            status_text = f"""Session Status: {session_status['state'].upper()}
Total Cycles: {stats['total_cycles']}
Success Rate: {stats['success_rate']:.1%}
Cycles/Min: {stats['cycles_per_minute']:.1f}
Errors: {len(stats['errors'])}"""
            
            self.health_axes['session_status'].text(0.1, 0.9, status_text,
                                                   transform=self.health_axes['session_status'].transAxes,
                                                   verticalalignment='top', fontfamily='monospace',
                                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            self.health_axes['session_status'].set_title('Session Status')
            self.health_axes['session_status'].axis('off')
            
        except Exception as e:
            print(f"Session status plot error: {e}")
    
    def _plot_cpu_memory_quick(self):
        """Plot quick CPU and memory view"""
        try:
            hardware_summary = self.hardware_monitor.get_current_summary()
            
            if 'error' not in hardware_summary:
                current_metrics = hardware_summary.get('current_metrics', {})
                cpu_data = current_metrics.get('cpu', {})
                memory_data = current_metrics.get('memory', {})
                
                # CPU usage over time (if we have history)
                cpu_history = self.hardware_monitor.get_history_data('cpu', 2)  # Last 2 minutes
                if cpu_history:
                    times = list(range(len(cpu_history)))
                    cpu_values = [entry.get('cpu_percent', 0) for entry in cpu_history]
                    
                    self.health_axes['cpu_memory'].plot(times, cpu_values, 'b-', label='CPU %', linewidth=2)
                    self.health_axes['cpu_memory'].set_ylabel('CPU Usage (%)', color='b')
                    self.health_axes['cpu_memory'].tick_params(axis='y', labelcolor='b')
                    
                    # Memory on secondary axis
                    ax2 = self.health_axes['cpu_memory'].twinx()
                    memory_history = self.hardware_monitor.get_history_data('memory', 2)
                    if memory_history:
                        memory_values = [entry.get('virtual_percent', 0) for entry in memory_history]
                        ax2.plot(times, memory_values, 'r-', label='Memory %', linewidth=2)
                        ax2.set_ylabel('Memory Usage (%)', color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                
                self.health_axes['cpu_memory'].set_title('CPU & Memory Trends')
                self.health_axes['cpu_memory'].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"CPU/Memory quick plot error: {e}")
    
    def _plot_disk_network_quick(self):
        """Plot quick disk and network view"""
        try:
            hardware_summary = self.hardware_monitor.get_current_summary()
            
            if 'error' not in hardware_summary:
                current_metrics = hardware_summary.get('current_metrics', {})
                disk_data = current_metrics.get('disk', {})
                network_data = current_metrics.get('network', {})
                
                # Disk and network rates
                disk_read_rate = disk_data.get('read_rate', 0) / 1024 / 1024  # MB/s
                disk_write_rate = disk_data.get('write_rate', 0) / 1024 / 1024  # MB/s
                net_recv_rate = network_data.get('bytes_recv_rate', 0) / 1024 / 1024  # MB/s
                net_sent_rate = network_data.get('bytes_sent_rate', 0) / 1024 / 1024  # MB/s
                
                categories = ['Disk Read', 'Disk Write', 'Net Recv', 'Net Sent']
                values = [disk_read_rate, disk_write_rate, net_recv_rate, net_sent_rate]
                colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
                
                bars = self.health_axes['disk_network'].bar(categories, values, color=colors, alpha=0.7)
                self.health_axes['disk_network'].set_title('Disk & Network I/O')
                self.health_axes['disk_network'].set_ylabel('Rate (MB/s)')
                
                # Rotate labels
                self.health_axes['disk_network'].tick_params(axis='x', rotation=45)
            
        except Exception as e:
            print(f"Disk/Network quick plot error: {e}")
    
    def _on_closing(self):
        """Handle window closing"""
        if self.is_monitoring:
            self._stop_monitoring()
        
        # Stop session if running
        if self.session_manager.state in [SessionState.RUNNING, SessionState.PAUSED]:
            self.session_manager.stop_session()
        
        self.root.destroy()
    
    def run(self):
        """Start the dashboard"""
        self.status_var.set("Dashboard ready - Click 'Start Monitoring' to begin")
        self.root.mainloop()
    
    def update_memory_traces(self, traces: List[MemoryTrace]):
        """Update memory traces for visualization"""
        self.memory_traces = traces
    
    def _initialize_thread_manager(self):
        """Initialize thread manager integration"""
        try:
            from .thread_manager import get_thread_manager
            self.thread_manager = get_thread_manager()
            
            # Add callbacks for thread events
            self.thread_manager.add_performance_callback(self._on_thread_performance_update)
            
            # Initialize thread history for visualization
            self.thread_history = {
                'timestamps': [],
                'active_counts': [],
                'cpu_usage': [],
                'memory_usage': [],
                'thread_types': {}
            }
            
            print("✅ Thread manager integration initialized")
            
        except ImportError:
            self.thread_manager = None
            print("⚠️ Thread manager not available")
    
    def _on_thread_performance_update(self, metrics: Dict[str, Any]):
        """Handle thread performance updates"""
        try:
            timestamp = datetime.now()
            
            # Update history
            self.thread_history['timestamps'].append(timestamp)
            self.thread_history['active_counts'].append(metrics.get('active_threads', 0))
            
            # Keep only recent history (last 100 points)
            if len(self.thread_history['timestamps']) > 100:
                for key in self.thread_history:
                    if isinstance(self.thread_history[key], list):
                        self.thread_history[key] = self.thread_history[key][-100:]
            
            # Update thread types history
            threads_by_type = metrics.get('threads_by_type', {})
            for thread_type, count in threads_by_type.items():
                if thread_type not in self.thread_history['thread_types']:
                    self.thread_history['thread_types'][thread_type] = []
                self.thread_history['thread_types'][thread_type].append(count)
                
                # Keep only recent history
                if len(self.thread_history['thread_types'][thread_type]) > 100:
                    self.thread_history['thread_types'][thread_type] = \
                        self.thread_history['thread_types'][thread_type][-100:]
            
        except Exception as e:
            print(f"Thread performance update error: {e}")
    
    def _refresh_thread_data(self):
        """Manually refresh thread data"""
        if self.thread_manager:
            try:
                summary = self.thread_manager.get_performance_summary()
                self._update_thread_stats_display(summary)
                print("🔄 Thread data refreshed")
            except Exception as e:
                print(f"Thread data refresh error: {e}")
    
    def _export_thread_data(self):
        """Export thread performance data"""
        if self.thread_manager:
            try:
                import json
                from tkinter import filedialog
                
                summary = self.thread_manager.get_performance_summary()
                
                filename = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Export Thread Data"
                )
                
                if filename:
                    with open(filename, 'w') as f:
                        json.dump(summary, f, indent=2, default=str)
                    print(f"📊 Thread data exported to: {filename}")
                    
            except Exception as e:
                print(f"Thread data export error: {e}")
    
    def _emergency_stop_threads(self):
        """Emergency stop all non-essential threads"""
        if self.thread_manager:
            try:
                from tkinter import messagebox
                
                result = messagebox.askyesno(
                    "Emergency Stop",
                    "This will terminate all non-essential threads. Continue?",
                    icon="warning"
                )
                
                if result:
                    self.thread_manager.emergency_terminate_all(exclude_types=['monitoring', 'dashboard'])
                    print("⚠️ Emergency thread termination initiated")
                    
            except Exception as e:
                print(f"Emergency stop error: {e}")
    
    def _update_thread_stats_display(self, summary: Dict[str, Any]):
        """Update thread statistics display"""
        try:
            thread_counts = summary.get('thread_counts', {})
            active = thread_counts.get('active', 0)
            peak = thread_counts.get('peak_concurrent', 0)
            total_created = thread_counts.get('total_created', 0)
            
            performance = summary.get('performance', {})
            avg_lifetime = performance.get('average_lifetime', 0)
            
            # Include thread pool information
            thread_pools = summary.get('thread_pools', {})
            total_pools = thread_pools.get('total_pools', 0)
            total_pool_workers = thread_pools.get('total_active_workers', 0)
            max_pool_workers = thread_pools.get('total_max_workers', 0)
            
            # Show more detailed information
            stats_text = (f"Active: {active} | Peak: {peak} | Created: {total_created} | "
                         f"Pools: {total_pools} | Pool Workers: {total_pool_workers}/{max_pool_workers} | "
                         f"Avg Life: {avg_lifetime:.1f}s")
            
            self.thread_stats_var.set(stats_text)
            
        except Exception as e:
            self.thread_stats_var.set(f"Thread stats error: {e}")
    
    def _update_thread_tab(self):
        """Update thread management tab - optimized for performance"""
        if not self.thread_manager or 'threads' not in self.figures:
            return
        
        try:
            # Rate limiting for thread tab
            if not hasattr(self, '_last_thread_update'):
                self._last_thread_update = 0
            
            current_time = time.time()
            if current_time - self._last_thread_update < 1.0:  # Update every 1 second max
                return
            
            self._last_thread_update = current_time
            
            summary = self.thread_manager.get_performance_summary()
            
            # Update stats display (lightweight)
            self._update_thread_stats_display(summary)
            
            # Clear only essential axes to reduce work
            essential_axes = ['thread_count', 'thread_types', 'thread_pools', 'pool_utilization']
            for ax_name in essential_axes:
                if ax_name in self.thread_axes:
                    self.thread_axes[ax_name].clear()
            
            # Plot thread count over time
            if len(self.thread_history['timestamps']) > 1:
                times = self.thread_history['timestamps']
                counts = self.thread_history['active_counts']
                
                self.thread_axes['thread_count'].plot(times, counts, 'b-', linewidth=2)
                self.thread_axes['thread_count'].set_title('Active Thread Count')
                self.thread_axes['thread_count'].set_ylabel('Count')
                self.thread_axes['thread_count'].tick_params(axis='x', rotation=45)
            
            # Plot threads by type (pie chart)
            threads_by_type = summary.get('threads_by_type', {})
            if threads_by_type:
                labels = list(threads_by_type.keys())
                sizes = list(threads_by_type.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                self.thread_axes['thread_types'].pie(sizes, labels=labels, colors=colors, 
                                                   autopct='%1.1f%%', startangle=90)
                self.thread_axes['thread_types'].set_title('Threads by Type')
            
            # Plot thread states
            current_threads = summary.get('current_threads', [])
            if current_threads:
                states = [t.get('state', 'unknown') for t in current_threads]
                state_counts = {}
                for state in states:
                    state_counts[state] = state_counts.get(state, 0) + 1
                
                if state_counts:
                    labels = list(state_counts.keys())
                    counts = list(state_counts.values())
                    colors = ['green' if s == 'running' else 'orange' if s == 'waiting' else 'red' 
                             for s in labels]
                    
                    self.thread_axes['thread_states'].bar(labels, counts, color=colors, alpha=0.7)
                    self.thread_axes['thread_states'].set_title('Thread States')
                    self.thread_axes['thread_states'].set_ylabel('Count')
            
            # Plot CPU usage by thread
            if current_threads:
                thread_names = [t.get('name', f"Thread-{t.get('id', '?')}")[:15] for t in current_threads[:10]]
                cpu_times = [t.get('cpu_time', 0) for t in current_threads[:10]]
                
                if any(cpu_times):
                    colors = ['red' if cpu > 1.0 else 'orange' if cpu > 0.5 else 'green' 
                             for cpu in cpu_times]
                    
                    bars = self.thread_axes['cpu_usage'].bar(range(len(thread_names)), cpu_times, 
                                                           color=colors, alpha=0.7)
                    self.thread_axes['cpu_usage'].set_title('Thread CPU Usage')
                    self.thread_axes['cpu_usage'].set_ylabel('CPU Time (s)')
                    self.thread_axes['cpu_usage'].set_xticks(range(len(thread_names)))
                    self.thread_axes['cpu_usage'].set_xticklabels(thread_names, rotation=45, ha='right')
            
            # Plot thread lifetimes
            if current_threads:
                lifetimes = [t.get('lifetime', 0) for t in current_threads]
                
                if lifetimes:
                    self.thread_axes['thread_lifetime'].hist(lifetimes, bins=10, alpha=0.7, color='skyblue')
                    self.thread_axes['thread_lifetime'].set_title('Thread Lifetimes')
                    self.thread_axes['thread_lifetime'].set_xlabel('Lifetime (seconds)')
                    self.thread_axes['thread_lifetime'].set_ylabel('Count')
            
            # Plot performance trends
            if len(self.thread_history['timestamps']) > 1:
                times = self.thread_history['timestamps']
                
                # Plot multiple thread types over time
                for thread_type, counts in self.thread_history['thread_types'].items():
                    if len(counts) == len(times):
                        self.thread_axes['performance_trends'].plot(times, counts, 
                                                                  label=thread_type, linewidth=2)
                
                self.thread_axes['performance_trends'].set_title('Thread Performance Trends')
                self.thread_axes['performance_trends'].set_ylabel('Thread Count')
                self.thread_axes['performance_trends'].legend()
                self.thread_axes['performance_trends'].tick_params(axis='x', rotation=45)
            
            # Plot resource limits status
            resource_limits = summary.get('resource_limits', {})
            thread_counts = summary.get('thread_counts', {})
            
            current_active = thread_counts.get('active', 0)
            max_threads = resource_limits.get('max_threads', 50)
            
            # Create gauge-like visualization for thread usage
            usage_percent = (current_active / max_threads) * 100 if max_threads > 0 else 0
            
            colors = ['green', 'yellow', 'red']
            sizes = [33.33, 33.33, 33.34]
            
            wedges, texts = self.thread_axes['resource_limits'].pie(sizes, colors=colors, 
                                                                  startangle=90, counterclock=False)
            
            # Add usage indicator
            angle = 90 - (usage_percent * 1.8)  # Convert to angle (180 degrees = 100%)
            self.thread_axes['resource_limits'].annotate('', xy=(0.8 * np.cos(np.radians(angle)), 
                                                               0.8 * np.sin(np.radians(angle))), 
                                                       xytext=(0, 0),
                                                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))
            
            self.thread_axes['resource_limits'].set_title(f'Thread Usage: {usage_percent:.1f}%')
            
            # Plot thread creation timeline
            if current_threads:
                creation_times = []
                thread_types = []
                
                for thread in current_threads:
                    # Simulate creation time based on lifetime
                    lifetime = thread.get('lifetime', 0)
                    creation_time = datetime.now() - timedelta(seconds=lifetime)
                    creation_times.append(creation_time)
                    thread_types.append(thread.get('type', 'unknown'))
                
                if creation_times:
                    # Create scatter plot
                    type_colors = {'session': 'red', 'monitoring': 'blue', 'dashboard': 'green', 
                                 'hardware': 'orange', 'tool': 'purple', 'custom': 'gray'}
                    
                    for i, (creation_time, thread_type) in enumerate(zip(creation_times, thread_types)):
                        color = type_colors.get(thread_type, 'gray')
                        self.thread_axes['thread_timeline'].scatter(creation_time, i, c=color, s=50, alpha=0.7)
                    
                    self.thread_axes['thread_timeline'].set_title('Thread Creation Timeline')
                    self.thread_axes['thread_timeline'].set_ylabel('Thread Index')
                    self.thread_axes['thread_timeline'].tick_params(axis='x', rotation=45)
            
            # Plot thread pools overview
            thread_pools = summary.get('thread_pools', {})
            if thread_pools and 'individual_pools' in thread_pools:
                pools = thread_pools['individual_pools']
                
                if pools:
                    pool_names = [info['name'][:10] for info in pools.values()]
                    max_workers = [info['max_workers'] for info in pools.values()]
                    active_workers = [info['active_count'] for info in pools.values()]
                    
                    x_pos = range(len(pool_names))
                    width = 0.35
                    
                    self.thread_axes['thread_pools'].bar([x - width/2 for x in x_pos], max_workers, 
                                                       width, label='Max Workers', alpha=0.7, color='lightblue')
                    self.thread_axes['thread_pools'].bar([x + width/2 for x in x_pos], active_workers, 
                                                       width, label='Active Workers', alpha=0.7, color='orange')
                    
                    self.thread_axes['thread_pools'].set_title('Thread Pools Overview')
                    self.thread_axes['thread_pools'].set_ylabel('Worker Count')
                    self.thread_axes['thread_pools'].set_xticks(x_pos)
                    self.thread_axes['thread_pools'].set_xticklabels(pool_names, rotation=45)
                    self.thread_axes['thread_pools'].legend()
            
            # Plot pool utilization
            if thread_pools and 'individual_pools' in thread_pools:
                pools = thread_pools['individual_pools']
                
                if pools:
                    pool_names = [info['name'][:10] for info in pools.values()]
                    utilizations = [info['utilization'] for info in pools.values()]
                    
                    colors = ['red' if u > 80 else 'orange' if u > 60 else 'green' for u in utilizations]
                    
                    bars = self.thread_axes['pool_utilization'].bar(pool_names, utilizations, 
                                                                  color=colors, alpha=0.7)
                    self.thread_axes['pool_utilization'].set_title('Pool Worker Utilization')
                    self.thread_axes['pool_utilization'].set_ylabel('Utilization (%)')
                    self.thread_axes['pool_utilization'].set_ylim(0, 100)
                    self.thread_axes['pool_utilization'].tick_params(axis='x', rotation=45)
                    
                    # Add utilization percentage labels on bars
                    for bar, util in zip(bars, utilizations):
                        height = bar.get_height()
                        self.thread_axes['pool_utilization'].text(bar.get_x() + bar.get_width()/2., height + 1,
                                                                f'{util:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Plot pool activity trends (placeholder for now)
            self.thread_axes['pool_activity'].clear()
            self.thread_axes['pool_activity'].text(0.5, 0.5, 'Pool Activity Trends\n(Historical data)',
                                                 ha='center', va='center',
                                                 transform=self.thread_axes['pool_activity'].transAxes)
            self.thread_axes['pool_activity'].set_title('Pool Activity Trends')
            
            # Refresh the canvas
            # Update canvas efficiently
            if 'threads' in self.canvases:
                self.canvases['threads'].draw_idle()
            
        except Exception as e:
            print(f"Thread tab update error: {e}")

    def _plot_cpu_usage(self, cpu_data: Dict[str, Any]):
        """Plot CPU usage"""
        try:
            cpu_percent = cpu_data.get('cpu_percent', 0)
            
            # Create gauge-like visualization
            angles = np.linspace(0, 2 * np.pi, 100)
            values = np.ones(100) * cpu_percent
            
            self.hardware_axes['cpu_usage'].fill_between(angles, 0, values, alpha=0.3, color='blue')
            self.hardware_axes['cpu_usage'].set_ylim(0, 100)
            self.hardware_axes['cpu_usage'].set_title(f'CPU Usage: {cpu_percent:.1f}%')
            
            # Add color coding
            color = 'red' if cpu_percent > 80 else 'orange' if cpu_percent > 60 else 'green'
            self.hardware_axes['cpu_usage'].text(0.5, 0.5, f'{cpu_percent:.1f}%',
                                               ha='center', va='center', fontsize=16, color=color,
                                               transform=self.hardware_axes['cpu_usage'].transAxes)
        except Exception as e:
            print(f"CPU usage plot error: {e}")
    
    def _plot_memory_usage(self, memory_data: Dict[str, Any]):
        """Plot memory usage"""
        try:
            virtual_percent = memory_data.get('virtual_percent', 0)
            virtual_used = memory_data.get('virtual_used', 0) / 1024**3  # GB
            virtual_total = memory_data.get('virtual_total', 0) / 1024**3  # GB
            
            # Pie chart
            used_gb = virtual_used
            free_gb = virtual_total - virtual_used
            
            sizes = [used_gb, free_gb]
            labels = ['Used', 'Free']
            colors = ['lightcoral', 'lightgreen']
            
            wedges, texts, autotexts = self.hardware_axes['memory_usage'].pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90
            )
            
            self.hardware_axes['memory_usage'].set_title(f'Memory: {used_gb:.1f}/{virtual_total:.1f} GB')
        except Exception as e:
            print(f"Memory usage plot error: {e}")
    
    def _plot_disk_usage(self, disk_data: Dict[str, Any]):
        """Plot disk usage"""
        try:
            disk_usage = disk_data.get('disk_usage', [])
            
            if disk_usage:
                # Show main disk partition
                main_disk = disk_usage[0]
                used_gb = main_disk['used'] / 1024**3
                total_gb = main_disk['total'] / 1024**3
                percent = main_disk['percent']
                
                # Bar chart
                categories = ['Used', 'Free']
                values = [used_gb, total_gb - used_gb]
                colors = ['lightcoral', 'lightgreen']
                
                bars = self.hardware_axes['disk_usage'].bar(categories, values, color=colors, alpha=0.7)
                self.hardware_axes['disk_usage'].set_title(f'Disk: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%)')
                self.hardware_axes['disk_usage'].set_ylabel('Storage (GB)')
            else:
                self.hardware_axes['disk_usage'].text(0.5, 0.5, 'No disk data',
                                                    ha='center', va='center',
                                                    transform=self.hardware_axes['disk_usage'].transAxes)
        except Exception as e:
            print(f"Disk usage plot error: {e}")
    
    def _plot_cpu_cores(self, cpu_data: Dict[str, Any]):
        """Plot CPU core usage"""
        try:
            cpu_per_core = cpu_data.get('cpu_per_core', [])
            
            if cpu_per_core:
                cores = list(range(len(cpu_per_core)))
                colors = ['red' if usage > 80 else 'orange' if usage > 60 else 'green' 
                         for usage in cpu_per_core]
                
                bars = self.hardware_axes['cpu_cores'].bar(cores, cpu_per_core, color=colors, alpha=0.7)
                self.hardware_axes['cpu_cores'].set_title('CPU Core Usage')
                self.hardware_axes['cpu_cores'].set_xlabel('Core')
                self.hardware_axes['cpu_cores'].set_ylabel('Usage (%)')
                self.hardware_axes['cpu_cores'].set_ylim(0, 100)
            else:
                self.hardware_axes['cpu_cores'].text(0.5, 0.5, 'No core data',
                                                   ha='center', va='center',
                                                   transform=self.hardware_axes['cpu_cores'].transAxes)
        except Exception as e:
            print(f"CPU cores plot error: {e}")
    
    def _plot_memory_details(self, memory_data: Dict[str, Any]):
        """Plot detailed memory information"""
        try:
            # Memory breakdown
            virtual_used = memory_data.get('virtual_used', 0) / 1024**3
            virtual_available = memory_data.get('virtual_available', 0) / 1024**3
            swap_used = memory_data.get('swap_used', 0) / 1024**3
            swap_total = memory_data.get('swap_total', 0) / 1024**3
            
            categories = ['Virtual Used', 'Virtual Available', 'Swap Used', 'Swap Total']
            values = [virtual_used, virtual_available, swap_used, swap_total]
            colors = ['lightcoral', 'lightgreen', 'lightblue', 'lightyellow']
            
            bars = self.hardware_axes['memory_details'].bar(categories, values, color=colors, alpha=0.7)
            self.hardware_axes['memory_details'].set_title('Memory Details')
            self.hardware_axes['memory_details'].set_ylabel('Memory (GB)')
            self.hardware_axes['memory_details'].tick_params(axis='x', rotation=45)
        except Exception as e:
            print(f"Memory details plot error: {e}")
    
    def _plot_disk_io(self, disk_data: Dict[str, Any]):
        """Plot disk I/O rates"""
        try:
            read_bytes = disk_data.get('read_bytes', 0) / 1024**2  # MB
            write_bytes = disk_data.get('write_bytes', 0) / 1024**2  # MB
            
            categories = ['Read', 'Write']
            values = [read_bytes, write_bytes]
            colors = ['lightblue', 'lightcoral']
            
            bars = self.hardware_axes['disk_io'].bar(categories, values, color=colors, alpha=0.7)
            self.hardware_axes['disk_io'].set_title('Disk I/O')
            self.hardware_axes['disk_io'].set_ylabel('Rate (MB/s)')
        except Exception as e:
            print(f"Disk I/O plot error: {e}")
    
    def _plot_network_io(self, network_data: Dict[str, Any]):
        """Plot network I/O rates"""
        try:
            bytes_sent = network_data.get('bytes_sent', 0) / 1024**2  # MB
            bytes_recv = network_data.get('bytes_recv', 0) / 1024**2  # MB
            
            categories = ['Sent', 'Received']
            values = [bytes_sent, bytes_recv]
            colors = ['lightgreen', 'lightblue']
            
            bars = self.hardware_axes['network_io'].bar(categories, values, color=colors, alpha=0.7)
            self.hardware_axes['network_io'].set_title('Network I/O')
            self.hardware_axes['network_io'].set_ylabel('Rate (MB/s)')
        except Exception as e:
            print(f"Network I/O plot error: {e}")
    
    def _plot_gpu_usage(self, gpu_data: Dict[str, Any]):
        """Plot GPU usage"""
        try:
            gpu_info = gpu_data.get('gpu_info', [])
            
            if gpu_info:
                # Show first GPU
                gpu = gpu_info[0]
                gpu_percent = gpu.get('load', 0)
                memory_used = gpu.get('memory_used', 0)
                memory_total = gpu.get('memory_total', 1)
                memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                
                categories = ['GPU Load', 'GPU Memory']
                values = [gpu_percent, memory_percent]
                colors = ['purple', 'orange']
                
                bars = self.hardware_axes['gpu_usage'].bar(categories, values, color=colors, alpha=0.7)
                self.hardware_axes['gpu_usage'].set_title(f'GPU: {gpu.get("name", "Unknown")}')
                self.hardware_axes['gpu_usage'].set_ylabel('Usage (%)')
                self.hardware_axes['gpu_usage'].set_ylim(0, 100)
            else:
                self.hardware_axes['gpu_usage'].text(0.5, 0.5, 'No GPU data',
                                                   ha='center', va='center',
                                                   transform=self.hardware_axes['gpu_usage'].transAxes)
                self.hardware_axes['gpu_usage'].set_title('GPU Usage')
        except Exception as e:
            print(f"GPU usage plot error: {e}")
    
    def _plot_temperatures(self, temp_data: Dict[str, Any]):
        """Plot system temperatures"""
        try:
            temperatures = temp_data.get('temperatures', {})
            
            if temperatures:
                labels = []
                temps = []
                colors = []
                
                for sensor, temp in temperatures.items():
                    if isinstance(temp, (int, float)):
                        labels.append(sensor[:10])  # Truncate long names
                        temps.append(temp)
                        # Color coding: green < 60, yellow < 80, red >= 80
                        if temp < 60:
                            colors.append('green')
                        elif temp < 80:
                            colors.append('orange')
                        else:
                            colors.append('red')
                
                if temps:
                    bars = self.hardware_axes['temperatures'].bar(labels, temps, color=colors, alpha=0.7)
                    self.hardware_axes['temperatures'].set_title('System Temperatures')
                    self.hardware_axes['temperatures'].set_ylabel('Temperature (°C)')
                    self.hardware_axes['temperatures'].tick_params(axis='x', rotation=45)
                    
                    # Add temperature threshold lines
                    self.hardware_axes['temperatures'].axhline(y=60, color='yellow', linestyle='--', alpha=0.5)
                    self.hardware_axes['temperatures'].axhline(y=80, color='red', linestyle='--', alpha=0.5)
            else:
                self.hardware_axes['temperatures'].text(0.5, 0.5, 'No temperature data',
                                                      ha='center', va='center',
                                                      transform=self.hardware_axes['temperatures'].transAxes)
                self.hardware_axes['temperatures'].set_title('System Temperatures')
        except Exception as e:
            print(f"Temperatures plot error: {e}")
    
    def _plot_process_info(self, process_data: Dict[str, Any]):
        """Plot process information"""
        try:
            process_count = process_data.get('process_count', 0)
            thread_count = process_data.get('thread_count', 0)
            handle_count = process_data.get('handle_count', 0)
            
            categories = ['Processes', 'Threads', 'Handles']
            values = [process_count, thread_count, handle_count]
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            
            bars = self.hardware_axes['process_info'].bar(categories, values, color=colors, alpha=0.7)
            self.hardware_axes['process_info'].set_title('Process Information')
            self.hardware_axes['process_info'].set_ylabel('Count')
        except Exception as e:
            print(f"Process info plot error: {e}")
    
    def _plot_system_info(self, system_info: Dict[str, Any]):
        """Plot system information"""
        try:
            # Display key system information as text
            info_text = []
            
            if 'platform' in system_info:
                info_text.append(f"Platform: {system_info['platform']}")
            if 'cpu_count' in system_info:
                info_text.append(f"CPU Cores: {system_info['cpu_count']}")
            if 'total_memory' in system_info:
                memory_gb = system_info['total_memory'] / 1024**3
                info_text.append(f"Total Memory: {memory_gb:.1f} GB")
            if 'boot_time' in system_info:
                info_text.append(f"Boot Time: {system_info['boot_time']}")
            
            # Clear and display text
            self.hardware_axes['system_info'].clear()
            self.hardware_axes['system_info'].axis('off')
            
            if info_text:
                text_content = '\n'.join(info_text)
                self.hardware_axes['system_info'].text(0.1, 0.9, text_content,
                                                     transform=self.hardware_axes['system_info'].transAxes,
                                                     fontsize=10, verticalalignment='top',
                                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            
            self.hardware_axes['system_info'].set_title('System Information')
        except Exception as e:
            print(f"System info plot error: {e}")
    
    def _plot_resource_trends(self):
        """Plot resource usage trends"""
        try:
            # This would typically show historical data
            # For now, show a placeholder
            self.hardware_axes['resource_trends'].clear()
            self.hardware_axes['resource_trends'].text(0.5, 0.5, 'Resource trends\n(Historical data)',
                                                     ha='center', va='center',
                                                     transform=self.hardware_axes['resource_trends'].transAxes)
            self.hardware_axes['resource_trends'].set_title('Resource Trends')
        except Exception as e:
            print(f"Resource trends plot error: {e}")


def create_interactive_dashboard(collector: InstrumentationCollector, 
                               memory_traces: Optional[List[MemoryTrace]] = None,
                               update_interval: float = 2.0) -> InteractiveDashboard:
    """
    Create and return an interactive dashboard instance.
    
    Args:
        collector: InstrumentationCollector for monitoring data
        memory_traces: Optional list of memory traces
        update_interval: Update frequency in seconds
    
    Returns:
        InteractiveDashboard instance
    """
    return InteractiveDashboard(collector, memory_traces, update_interval)


if __name__ == "__main__":
    # Example usage
    from logging_config import InstrumentationCollector
    
    # Create mock collector for testing
    collector = InstrumentationCollector()
    
    # Add some mock data
    collector.metrics['cycle_timings'] = [
        {'cycle': i, 'total_duration': 0.5 + 0.1 * np.sin(i * 0.1)} 
        for i in range(1, 51)
    ]
    
    # Create and run dashboard
    dashboard = create_interactive_dashboard(collector)
    dashboard.run()
