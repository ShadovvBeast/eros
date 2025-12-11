"""
Session Control Panel

Comprehensive session management controls for the dashboard.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional

# Import with fallback for different execution contexts
try:
    from ..monitoring.session_manager import SessionManager, SessionConfig, SessionState
except ImportError:
    try:
        from monitoring.session_manager import SessionManager, SessionConfig, SessionState
    except ImportError:
        # Mock implementations for testing
        from enum import Enum
        
        class SessionState(Enum):
            IDLE = "idle"
            RUNNING = "running"
            PAUSED = "paused"
            COMPLETED = "completed"
            ERROR = "error"
        
        class SessionConfig:
            def __init__(self):
                self.duration_minutes = 5.0
                self.agent_identity = "Test Agent"
                self.pathos_dimension = 128
                self.exploration_factor = 0.1
                self.output_directory = "test_output"
        
        class SessionManager:
            def __init__(self):
                self.state = SessionState.IDLE
            def configure_session(self, config): pass
            def start_session(self): pass
            def stop_session(self): pass
            def pause_session(self): pass
            def resume_session(self): pass
            def emergency_stop(self): pass


class SessionControlPanel:
    """Session control panel with comprehensive management features."""
    
    def __init__(self, parent, session_manager: SessionManager, 
                 session_config: SessionConfig,
                 export_data_callback: Callable,
                 open_config_callback: Callable):
        """Initialize session control panel."""
        self.session_manager = session_manager
        self.session_config = session_config
        self.export_data_callback = export_data_callback
        self.open_config_callback = open_config_callback
        
        self._create_controls(parent)
    
    def _create_controls(self, parent):
        """Create comprehensive session control panel."""
        control_frame = ttk.LabelFrame(parent, text="Session Control", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - Session controls
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Session buttons
        session_frame = ttk.Frame(left_frame)
        session_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.start_button = ttk.Button(
            session_frame,
            text="▶️ Start Session",
            command=self._toggle_session,
            style="Accent.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_button = ttk.Button(
            session_frame,
            text="⏸️ Pause",
            command=self._pause_session,
            state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.emergency_button = ttk.Button(
            session_frame,
            text="🛑 Emergency Stop",
            command=self._emergency_stop,
            style="Danger.TButton"
        )
        self.emergency_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Configuration button
        ttk.Button(
            session_frame,
            text="⚙️ Configure Agent",
            command=self.open_config_dialog
        ).pack(side=tk.LEFT)
        
        # Right side - Monitoring controls
        right_frame = ttk.Frame(control_frame)
        right_frame.pack(side=tk.RIGHT)
        
        # Dashboard controls (no monitoring button - automatic with session)
        controls_frame = ttk.Frame(right_frame)
        controls_frame.pack(fill=tk.X)
        
        ttk.Button(
            controls_frame,
            text="🔄 Refresh",
            command=self._refresh_all
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            controls_frame,
            text="💾 Export Data",
            command=self.export_data_callback
        ).pack(side=tk.LEFT)
    
    def _toggle_session(self):
        """Toggle session start/stop."""
        if self.session_manager.state == SessionState.IDLE:
            self._start_session()
        else:
            self._stop_session()
    
    def _start_session(self):
        """Start agent session."""
        try:
            # Update button states
            self.start_button.config(text="⏹️ Stop Session", style="Danger.TButton")
            self.pause_button.config(state=tk.NORMAL)
            
            # Configure and start session
            self.session_manager.configure_session(self.session_config)
            self.session_manager.start_session()
            
        except Exception as e:
            messagebox.showerror("Session Error", f"Failed to start session: {e}")
            self._reset_button_states()
    
    def _stop_session(self):
        """Stop agent session."""
        self.session_manager.stop_session()
        self._reset_button_states()
    
    def _pause_session(self):
        """Pause/resume session."""
        if self.session_manager.state == SessionState.RUNNING:
            self.session_manager.pause_session()
            self.pause_button.config(text="▶️ Resume")
        elif self.session_manager.state == SessionState.PAUSED:
            self.session_manager.resume_session()
            self.pause_button.config(text="⏸️ Pause")
    
    def _emergency_stop(self):
        """Emergency stop with confirmation."""
        if messagebox.askyesno("Emergency Stop", 
                              "This will immediately terminate the session. Continue?"):
            self.session_manager.emergency_stop()
            self._reset_button_states()
    
    def _reset_button_states(self):
        """Reset button states to default."""
        self.start_button.config(text="▶️ Start Session", style="Accent.TButton")
        self.pause_button.config(text="⏸️ Pause", state=tk.DISABLED)
    
    def _refresh_all(self):
        """Refresh all displays."""
        # This would trigger refresh in parent dashboard
        pass
    
    def open_config_dialog(self):
        """Open agent configuration dialog."""
        config_window = tk.Toplevel()
        config_window.title("Agent Configuration")
        config_window.geometry("500x600")
        config_window.transient()
        config_window.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(config_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Agent Configuration", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 20))
        
        # Session Duration
        duration_frame = ttk.LabelFrame(main_frame, text="Session Settings", padding="10")
        duration_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(duration_frame, text="Duration (minutes):").pack(anchor=tk.W)
        duration_var = tk.DoubleVar(value=self.session_config.duration_minutes)
        duration_scale = ttk.Scale(duration_frame, from_=1, to=60, 
                                  variable=duration_var, orient=tk.HORIZONTAL)
        duration_scale.pack(fill=tk.X, pady=(5, 0))
        
        duration_label = ttk.Label(duration_frame, text=f"Current: {duration_var.get():.1f} minutes")
        duration_label.pack(anchor=tk.W, pady=(5, 0))
        
        def update_duration_label(*args):
            duration_label.config(text=f"Current: {duration_var.get():.1f} minutes")
        duration_var.trace('w', update_duration_label)
        
        # Agent Identity
        identity_frame = ttk.LabelFrame(main_frame, text="Agent Identity", padding="10")
        identity_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(identity_frame, text="Agent Name:").pack(anchor=tk.W)
        identity_var = tk.StringVar(value=self.session_config.agent_identity)
        ttk.Entry(identity_frame, textvariable=identity_var, width=40).pack(fill=tk.X, pady=(5, 0))
        
        # Pathos Configuration
        pathos_frame = ttk.LabelFrame(main_frame, text="Pathos Configuration", padding="10")
        pathos_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(pathos_frame, text="State Dimension:").pack(anchor=tk.W)
        pathos_dim_var = tk.IntVar(value=self.session_config.pathos_dimension)
        pathos_scale = ttk.Scale(pathos_frame, from_=32, to=512, 
                               variable=pathos_dim_var, orient=tk.HORIZONTAL)
        pathos_scale.pack(fill=tk.X, pady=(5, 0))
        
        pathos_label = ttk.Label(pathos_frame, text=f"Current: {pathos_dim_var.get()}")
        pathos_label.pack(anchor=tk.W, pady=(5, 0))
        
        def update_pathos_label(*args):
            pathos_label.config(text=f"Current: {pathos_dim_var.get()}")
        pathos_dim_var.trace('w', update_pathos_label)
        
        # Exploration Factor
        ttk.Label(pathos_frame, text="Exploration Factor:").pack(anchor=tk.W, pady=(10, 0))
        exploration_var = tk.DoubleVar(value=self.session_config.exploration_factor)
        exploration_scale = ttk.Scale(pathos_frame, from_=0.01, to=1.0, 
                                    variable=exploration_var, orient=tk.HORIZONTAL)
        exploration_scale.pack(fill=tk.X, pady=(5, 0))
        
        exploration_label = ttk.Label(pathos_frame, text=f"Current: {exploration_var.get():.2f}")
        exploration_label.pack(anchor=tk.W, pady=(5, 0))
        
        def update_exploration_label(*args):
            exploration_label.config(text=f"Current: {exploration_var.get():.2f}")
        exploration_var.trace('w', update_exploration_label)
        
        # Output Directory
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(output_frame, text="Output Directory:").pack(anchor=tk.W)
        output_var = tk.StringVar(value=self.session_config.output_directory)
        ttk.Entry(output_frame, textvariable=output_var, width=40).pack(fill=tk.X, pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def save_config():
            self.session_config.duration_minutes = duration_var.get()
            self.session_config.agent_identity = identity_var.get()
            self.session_config.pathos_dimension = pathos_dim_var.get()
            self.session_config.exploration_factor = exploration_var.get()
            self.session_config.output_directory = output_var.get()
            config_window.destroy()
            messagebox.showinfo("Configuration", "Agent configuration updated successfully!")
        
        ttk.Button(button_frame, text="Save", command=save_config).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=config_window.destroy).pack(side=tk.RIGHT)