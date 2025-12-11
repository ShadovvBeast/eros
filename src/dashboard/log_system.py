"""
Log and Audit System

Comprehensive logging and audit trail functionality for the dashboard.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque
import json
import os

try:
    from ..monitoring.session_manager import SessionManager, SessionState
except ImportError:
    from monitoring.session_manager import SessionManager, SessionState


class LogAuditSystem:
    """Comprehensive log and audit trail system."""
    
    def __init__(self, notebook, session_manager: SessionManager):
        """Initialize log audit system."""
        self.session_manager = session_manager
        self.is_monitoring = False
        self.log_thread = None
        
        # Log storage
        self.log_buffer = deque(maxlen=10000)  # Store up to 10k entries
        self.log_statistics = {
            'total_entries': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'debug_count': 0
        }
        
        # UI components
        self.log_text = None
        self.search_var = tk.StringVar()
        self.level_filter_var = tk.StringVar(value="ALL")
        self.component_filter_var = tk.StringVar(value="ALL")
        self.auto_scroll_var = tk.BooleanVar(value=True)
        self._search_timer = None
        
        # Create the tab
        self._create_tab(notebook)
        self._setup_log_capture()
    
    def _create_tab(self, notebook):
        """Create comprehensive log and audit trail tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="📋 Log & Audit")
        
        # Top controls
        controls_frame = ttk.Frame(frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Statistics display
        self._create_log_statistics_display(controls_frame)
        
        # Filter controls
        filter_frame = ttk.Frame(frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        self._create_log_filter_controls(filter_frame)
        
        # Log display
        log_display_frame = ttk.Frame(frame)
        log_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(log_display_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags for different log levels with enhanced styling
        self.log_text.tag_configure("ERROR", foreground="#FF4444", font=('Consolas', 9, 'bold'), background="#FFE6E6")
        self.log_text.tag_configure("WARNING", foreground="#FF8800", font=('Consolas', 9, 'bold'), background="#FFF4E6")
        self.log_text.tag_configure("INFO", foreground="#0066CC", font=('Consolas', 9))
        self.log_text.tag_configure("DEBUG", foreground="#666666", font=('Consolas', 8))
        self.log_text.tag_configure("TIMESTAMP", foreground="#8B4B9B", font=('Consolas', 8, 'italic'))
        self.log_text.tag_configure("COMPONENT", foreground="#228B22", font=('Consolas', 9, 'bold'))
        
        # Enhanced component-specific styling
        self.log_text.tag_configure("LOGOS", foreground="#4169E1", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("PATHOS", foreground="#DC143C", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("MEMORY", foreground="#FF6347", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("TOOLS", foreground="#32CD32", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("AGENT", foreground="#1E90FF", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("SESSION", foreground="#9932CC", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("SYSTEM", foreground="#696969", font=('Consolas', 9, 'bold'))
        
        # Special formatting for cycle summaries
        self.log_text.tag_configure("CYCLE_HEADER", foreground="#000080", font=('Consolas', 10, 'bold'), background="#F0F8FF")
        self.log_text.tag_configure("DETAILS", foreground="#2F4F4F", font=('Consolas', 8))
        self.log_text.tag_configure("SEPARATOR", foreground="#CCCCCC")
        
        # Export controls
        export_frame = ttk.Frame(frame)
        export_frame.pack(fill=tk.X, pady=(10, 0))
        self._create_log_export_controls(export_frame)
    
    def _create_log_statistics_display(self, parent):
        """Create log statistics display."""
        stats_frame = ttk.LabelFrame(parent, text="Log Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Statistics grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Total entries
        ttk.Label(stats_grid, text="Total:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.total_label = ttk.Label(stats_grid, text="0", font=('Arial', 9, 'bold'))
        self.total_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        
        # Error count
        ttk.Label(stats_grid, text="Errors:", foreground="red").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.error_label = ttk.Label(stats_grid, text="0", foreground="red", font=('Arial', 9, 'bold'))
        self.error_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 15))
        
        # Warning count
        ttk.Label(stats_grid, text="Warnings:", foreground="orange").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.warning_label = ttk.Label(stats_grid, text="0", foreground="orange", font=('Arial', 9, 'bold'))
        self.warning_label.grid(row=0, column=5, sticky=tk.W, padx=(0, 15))
        
        # Info count
        ttk.Label(stats_grid, text="Info:", foreground="blue").grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        self.info_label = ttk.Label(stats_grid, text="0", foreground="blue", font=('Arial', 9, 'bold'))
        self.info_label.grid(row=0, column=7, sticky=tk.W, padx=(0, 15))
        
        # Last update
        ttk.Label(stats_grid, text="Last Update:").grid(row=0, column=8, sticky=tk.W, padx=(0, 5))
        self.last_update_label = ttk.Label(stats_grid, text="Never", font=('Arial', 9))
        self.last_update_label.grid(row=0, column=9, sticky=tk.W)
    
    def _create_log_filter_controls(self, parent):
        """Create log filtering controls."""
        # Log level filter
        level_frame = ttk.Frame(parent)
        level_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(level_frame, text="Level:").pack(side=tk.LEFT)
        level_combo = ttk.Combobox(level_frame, textvariable=self.level_filter_var,
                                  values=["ALL", "ERROR", "WARNING", "INFO", "DEBUG"],
                                  state="readonly", width=10)
        level_combo.pack(side=tk.LEFT, padx=(5, 0))
        level_combo.bind('<<ComboboxSelected>>', lambda e: self._update_log_filter())
        
        # Component filter
        component_frame = ttk.Frame(parent)
        component_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(component_frame, text="Component:").pack(side=tk.LEFT)
        component_combo = ttk.Combobox(component_frame, textvariable=self.component_filter_var,
                                     values=["ALL", "SESSION", "AGENT", "PATHOS", "MEMORY", "LOGOS", "TOOLS", "SYSTEM"],
                                     state="readonly", width=12)
        component_combo.pack(side=tk.LEFT, padx=(5, 0))
        component_combo.bind('<<ComboboxSelected>>', lambda e: self._update_log_filter())
        
        # Detailed logging toggle
        detail_frame = ttk.Frame(parent)
        detail_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.detailed_logging_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(detail_frame, text="Detailed Cycle Logging", 
                       variable=self.detailed_logging_var,
                       command=self._toggle_detailed_logging).pack(side=tk.LEFT)
        
        # Search
        search_frame = ttk.Frame(parent)
        search_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=(5, 0))
        search_entry.bind('<KeyRelease>', self._on_search_change)
        
        ttk.Button(search_frame, text="Clear", command=self._clear_search).pack(side=tk.LEFT, padx=(5, 0))
        
        # Clear log button
        ttk.Button(parent, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)
    
    def _create_log_export_controls(self, parent):
        """Create log export and control options."""
        # Auto-scroll control
        ttk.Checkbutton(parent, text="Auto-scroll", variable=self.auto_scroll_var).pack(side=tk.LEFT)
        
        # Export buttons
        export_buttons = ttk.Frame(parent)
        export_buttons.pack(side=tk.RIGHT)
        
        ttk.Button(export_buttons, text="Export Current", 
                  command=self._export_current_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_buttons, text="Export Statistics", 
                  command=self._export_log_statistics).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_buttons, text="Export Filtered", 
                  command=self._export_filtered_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_buttons, text="Copy to Clipboard", 
                  command=self._copy_log_to_clipboard).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_buttons, text="Refresh", 
                  command=self._refresh_log_display).pack(side=tk.LEFT)
    
    def start_monitoring(self):
        """Start log monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.log_thread = threading.Thread(target=self._log_update_loop, daemon=True)
            self.log_thread.start()
            self._add_log_entry("SYSTEM", "INFO", "Log monitoring started")
    
    def stop_monitoring(self):
        """Stop log monitoring."""
        if self.is_monitoring:
            self.is_monitoring = False
            self._add_log_entry("SYSTEM", "INFO", "Log monitoring stopped")
    
    def _setup_log_capture(self):
        """Setup log capture from system components."""
        # Capture from session manager
        self.session_manager.add_state_change_callback(self._on_session_event)
        self.session_manager.add_cycle_complete_callback(self._on_cycle_event)
        self.session_manager.add_error_callback(self._on_error_event)
        
        # Setup custom log handler to capture agent debug messages
        self._setup_agent_log_handler()
        
        self._add_log_entry("SYSTEM", "INFO", "Log & Audit system started")
    
    def _setup_agent_log_handler(self):
        """Setup custom handler to capture agent log messages."""
        import logging
        
        class DashboardLogHandler(logging.Handler):
            def __init__(self, dashboard_log_system):
                super().__init__()
                self.dashboard = dashboard_log_system
            
            def emit(self, record):
                try:
                    # Parse the log record
                    message = record.getMessage()
                    level = record.levelname
                    
                    # Determine component from logger name or message content
                    component = "AGENT"
                    if "logos" in message.lower():
                        component = "LOGOS"
                    elif "pathos" in message.lower():
                        component = "PATHOS"
                    elif "memory" in message.lower():
                        component = "MEMORY"
                    elif "tool" in message.lower():
                        component = "TOOLS"
                    
                    # Only show debug messages if detailed logging is enabled
                    if level == "DEBUG" and not self.dashboard.detailed_logging_var.get():
                        return
                    
                    # Extract details from structured logging
                    details = ""
                    if " | " in message:
                        # Split message and details (our logger format: "message | {json_data}")
                        parts = message.split(" | ", 1)
                        if len(parts) == 2:
                            message = parts[0]
                            try:
                                import json
                                kwargs_data = json.loads(parts[1])
                                if isinstance(kwargs_data, dict):
                                    details_parts = []
                                    for key, value in kwargs_data.items():
                                        if isinstance(value, float):
                                            details_parts.append(f"{key}: {value:.3f}")
                                        else:
                                            details_parts.append(f"{key}: {str(value)[:50]}")
                                    details = " | ".join(details_parts)
                            except (json.JSONDecodeError, ValueError):
                                # If JSON parsing fails, use the raw details
                                details = parts[1]
                    
                    self.dashboard._add_log_entry(component, level, message, details)
                except Exception:
                    pass  # Avoid recursive logging errors
        
        # Add handler to agent logger
        agent_logger = logging.getLogger('autonomous_agent')
        self.agent_log_handler = DashboardLogHandler(self)
        agent_logger.addHandler(self.agent_log_handler)
    
    def _log_update_loop(self):
        """Main loop for updating log display."""
        while self.is_monitoring:
            try:
                # Update statistics
                self._update_log_statistics()
                time.sleep(1)
            except Exception as e:
                print(f"Log update error: {e}")
                time.sleep(1)
    
    def _add_log_entry(self, component: str, level: str, message: str, details: str = ""):
        """Add a new log entry to the system."""
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
        
        # Update statistics
        self.log_statistics['total_entries'] += 1
        level_key = f"{level.lower()}_count"
        if level_key in self.log_statistics:
            self.log_statistics[level_key] += 1
        
        # Update display (thread-safe)
        if hasattr(self, 'log_text') and self.log_text:
            # Schedule UI update on main thread
            self.log_text.after(0, lambda: self._display_log_entry(log_entry))
    
    def _display_log_entry(self, entry):
        """Display a log entry in the text widget with enhanced formatting."""
        if not hasattr(self, 'log_text') or not self.log_text:
            return
        
        # Check if entry passes filters
        if not self._passes_filters(entry):
            return
        
        # Skip if this is a cycle completion (handled by _add_enhanced_cycle_entry)
        if "Cycle" in entry['message'] and "completed" in entry['message']:
            return
        
        # Format timestamp
        timestamp_str = entry['timestamp'].strftime("%H:%M:%S.%f")[:-3]
        
        # Determine component tag for enhanced styling
        component_tag = entry['component'] if entry['component'] in ['LOGOS', 'PATHOS', 'MEMORY', 'TOOLS', 'AGENT', 'SESSION', 'SYSTEM'] else "COMPONENT"
        
        # Insert entry with enhanced formatting
        self.log_text.insert(tk.END, f"[{timestamp_str}] ", "TIMESTAMP")
        self.log_text.insert(tk.END, f"[{entry['component']}] ", component_tag)
        self.log_text.insert(tk.END, f"[{entry['level']}] ", entry['level'])
        
        # Enhanced message formatting for debug entries
        if entry['level'] == 'DEBUG' and entry['details']:
            # Format debug messages with structured details
            self.log_text.insert(tk.END, f"{entry['message']}\n", entry['level'])
            
            # Parse and format details nicely
            details = entry['details']
            if '|' in details:
                detail_parts = [part.strip() for part in details.split('|')]
                for i, part in enumerate(detail_parts):
                    if ':' in part:
                        key, value = part.split(':', 1)
                        self.log_text.insert(tk.END, f"    • {key.strip()}: ", "DETAILS")
                        self.log_text.insert(tk.END, f"{value.strip()}\n", "DEBUG")
                    else:
                        self.log_text.insert(tk.END, f"    • {part}\n", "DEBUG")
            else:
                self.log_text.insert(tk.END, f"    {details}\n", "DETAILS")
        else:
            # Standard formatting for non-debug messages
            self.log_text.insert(tk.END, f"{entry['message']}")
            if entry['details']:
                self.log_text.insert(tk.END, f" - {entry['details']}")
            self.log_text.insert(tk.END, "\n")
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)
        
        # Limit display size (keep last 1000 lines)
        lines = int(self.log_text.index(tk.END).split('.')[0])
        if lines > 1000:
            excess_lines = lines - 1000
            self.log_text.delete("1.0", f"{excess_lines + 1}.0")
    
    def _passes_filters(self, entry):
        """Check if log entry passes current filters."""
        # Level filter
        if self.level_filter_var.get() != "ALL" and entry['level'] != self.level_filter_var.get():
            return False
        
        # Component filter
        if self.component_filter_var.get() != "ALL" and entry['component'] != self.component_filter_var.get():
            return False
        
        # Search filter
        search_text = self.search_var.get().lower()
        if search_text:
            searchable_text = f"{entry['message']} {entry['details']}".lower()
            if search_text not in searchable_text:
                return False
        
        return True
    
    def _update_log_statistics(self):
        """Update log statistics display."""
        try:
            if hasattr(self, 'total_label'):
                self.total_label.config(text=str(self.log_statistics['total_entries']))
                self.error_label.config(text=str(self.log_statistics['error_count']))
                self.warning_label.config(text=str(self.log_statistics['warning_count']))
                self.info_label.config(text=str(self.log_statistics['info_count']))
                self.last_update_label.config(text=datetime.now().strftime("%H:%M:%S"))
        except Exception as e:
            print(f"Statistics update error: {e}")
    
    # Event handlers
    def _on_session_event(self, state):
        """Handle session state change events."""
        self._add_log_entry("SESSION", "INFO", f"State changed to: {state.value}")
    
    def _on_cycle_event(self, cycle, data):
        """Handle agent cycle completion events with enhanced visual formatting."""
        duration = data.get('duration', 0)
        intention = data.get('intention', 'Unknown')
        semantic_category = data.get('semantic_category', 'Unknown')
        internal_reward = data.get('internal_reward', 0.0)
        external_reward = data.get('external_reward', 0.0)
        salience = data.get('salience', 0.0)
        memory_stored = data.get('memory_stored', False)
        tool_used = data.get('tool_used', None)
        
        # Create visually appealing cycle summary
        self._add_enhanced_cycle_entry(cycle, {
            'duration': duration,
            'intention': intention,
            'semantic_category': semantic_category,
            'internal_reward': internal_reward,
            'external_reward': external_reward,
            'salience': salience,
            'memory_stored': memory_stored,
            'tool_used': tool_used
        })
    
    def _add_enhanced_cycle_entry(self, cycle, data):
        """Add a beautifully formatted cycle entry to the log."""
        if not hasattr(self, 'log_text') or not self.log_text:
            return
        
        timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Cycle header with visual separator
        self.log_text.insert(tk.END, "\n" + "─" * 80 + "\n", "SEPARATOR")
        self.log_text.insert(tk.END, f"[{timestamp_str}] ", "TIMESTAMP")
        self.log_text.insert(tk.END, f"🔄 CYCLE {cycle} COMPLETED ", "CYCLE_HEADER")
        
        # Performance indicator
        duration = data['duration']
        if duration < 0.01:
            perf_icon = "⚡"
            perf_color = "INFO"
        elif duration < 0.1:
            perf_icon = "🟢"
            perf_color = "INFO"
        elif duration < 1.0:
            perf_icon = "🟡"
            perf_color = "WARNING"
        else:
            perf_icon = "🔴"
            perf_color = "ERROR"
        
        self.log_text.insert(tk.END, f"{perf_icon} {duration:.3f}s\n", perf_color)
        
        # Intention and category
        intention = data['intention']
        category = data['semantic_category']
        self.log_text.insert(tk.END, "  💭 Intention: ", "DETAILS")
        self.log_text.insert(tk.END, f"{intention[:70]}{'...' if len(intention) > 70 else ''}\n", "INFO")
        self.log_text.insert(tk.END, f"  🏷️  Category: ", "DETAILS")
        self.log_text.insert(tk.END, f"{category.upper()}\n", "COMPONENT")
        
        # Rewards section
        internal_reward = data['internal_reward']
        external_reward = data['external_reward']
        total_reward = internal_reward + external_reward
        
        reward_icon = "🎯" if total_reward > 0.5 else "📈" if total_reward > 0 else "📉" if total_reward < 0 else "➖"
        reward_color = "INFO" if total_reward > 0 else "WARNING" if total_reward < 0 else "DEBUG"
        
        self.log_text.insert(tk.END, f"  {reward_icon} Rewards: ", "DETAILS")
        self.log_text.insert(tk.END, f"Internal: {internal_reward:+.3f} | External: {external_reward:+.3f} | Total: {total_reward:+.3f}\n", reward_color)
        
        # Memory and salience
        salience = data['salience']
        memory_stored = data['memory_stored']
        memory_icon = "💾" if memory_stored else "🗃️"
        memory_color = "INFO" if memory_stored else "DEBUG"
        
        self.log_text.insert(tk.END, f"  {memory_icon} Memory: ", "DETAILS")
        self.log_text.insert(tk.END, f"{'STORED' if memory_stored else 'NOT STORED'} ", memory_color)
        self.log_text.insert(tk.END, f"(Salience: {salience:.3f})\n", "DETAILS")
        
        # Tool usage
        tool_used = data['tool_used']
        if tool_used:
            self.log_text.insert(tk.END, "  🔧 Tool Used: ", "DETAILS")
            self.log_text.insert(tk.END, f"{tool_used}\n", "TOOLS")
        else:
            self.log_text.insert(tk.END, "  🔧 Tool Used: ", "DETAILS")
            self.log_text.insert(tk.END, "None\n", "DEBUG")
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.log_text.see(tk.END)
    
    def _on_error_event(self, error):
        """Handle error events."""
        self._add_log_entry("SYSTEM", "ERROR", f"Error occurred: {str(error)}")
    
    # Filter and control methods
    def _update_log_filter(self):
        """Update log display based on current filters."""
        # Clear current display
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.delete("1.0", tk.END)
            
            # Redisplay filtered entries
            for entry in self.log_buffer:
                if self._passes_filters(entry):
                    self._display_log_entry(entry)
    
    def _on_search_change(self, event=None):
        """Handle search text changes."""
        # Debounce search updates
        if self._search_timer:
            self.log_text.after_cancel(self._search_timer)
        
        self._search_timer = self.log_text.after(500, self._update_log_filter)
    
    def _clear_search(self):
        """Clear search filter."""
        self.search_var.set("")
        self._update_log_filter()
    
    def _clear_log(self):
        """Clear all log entries."""
        if messagebox.askyesno("Clear Log", "Are you sure you want to clear all log entries?"):
            self.log_buffer.clear()
            self.log_statistics = {key: 0 for key in self.log_statistics}
            if hasattr(self, 'log_text') and self.log_text:
                self.log_text.delete("1.0", tk.END)
            self._add_log_entry("SYSTEM", "INFO", "Log cleared by user")
    
    # Export methods
    def _export_current_log(self):
        """Export current log to file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Current Log"
            )
            if filename:
                with open(filename, 'w') as f:
                    for entry in self.log_buffer:
                        timestamp_str = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        f.write(f"[{timestamp_str}] [{entry['component']}] [{entry['level']}] {entry['message']}")
                        if entry['details']:
                            f.write(f" - {entry['details']}")
                        f.write("\n")
                self._add_log_entry("SYSTEM", "INFO", f"Log exported to {filename}")
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Export failed: {str(e)}")
    
    def _export_log_statistics(self):
        """Export log statistics to JSON."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Log Statistics"
            )
            if filename:
                stats_data = {
                    'export_time': datetime.now().isoformat(),
                    'statistics': self.log_statistics,
                    'total_buffer_size': len(self.log_buffer),
                    'session_state': self.session_manager.state.value if self.session_manager else "Unknown"
                }
                
                with open(filename, 'w') as f:
                    json.dump(stats_data, f, indent=2)
                
                self._add_log_entry("SYSTEM", "INFO", f"Statistics exported to {filename}")
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Statistics export failed: {str(e)}")
    
    def _export_filtered_log(self):
        """Export only filtered log entries."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Filtered Log"
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(f"# Filtered Log Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Level Filter: {self.level_filter_var.get()}\n")
                    f.write(f"# Component Filter: {self.component_filter_var.get()}\n")
                    f.write(f"# Search Filter: '{self.search_var.get()}'\n\n")
                    
                    for entry in self.log_buffer:
                        if self._passes_filters(entry):
                            timestamp_str = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            f.write(f"[{timestamp_str}] [{entry['component']}] [{entry['level']}] {entry['message']}")
                            if entry['details']:
                                f.write(f" - {entry['details']}")
                            f.write("\n")
                
                self._add_log_entry("SYSTEM", "INFO", f"Filtered log exported to {filename}")
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Filtered export failed: {str(e)}")
    
    def _copy_log_to_clipboard(self):
        """Copy current log display to clipboard."""
        try:
            if hasattr(self, 'log_text') and self.log_text:
                content = self.log_text.get("1.0", tk.END)
                self.log_text.clipboard_clear()
                self.log_text.clipboard_append(content)
                self._add_log_entry("SYSTEM", "INFO", "Log copied to clipboard")
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Clipboard copy failed: {str(e)}")
    
    def _refresh_log_display(self):
        """Refresh the log display."""
        self._update_log_filter()
        self._add_log_entry("SYSTEM", "INFO", "Log display refreshed")
    
    def _toggle_detailed_logging(self):
        """Toggle detailed cycle logging."""
        if self.detailed_logging_var.get():
            self._add_log_entry("SYSTEM", "INFO", "Detailed cycle logging enabled")
            # Enable debug level logging for agent components
            import logging
            logging.getLogger('autonomous_agent').setLevel(logging.DEBUG)
        else:
            self._add_log_entry("SYSTEM", "INFO", "Detailed cycle logging disabled")
            # Reset to INFO level
            import logging
            logging.getLogger('autonomous_agent').setLevel(logging.INFO)
    
    def update_display(self):
        """Update display (called by dashboard monitoring loop)."""
        # Statistics are updated by the log thread
        pass
    
    def export_data(self, export_dir):
        """Export log data to directory."""
        try:
            # Export current log
            log_file = os.path.join(export_dir, "log_audit.txt")
            with open(log_file, 'w') as f:
                for entry in self.log_buffer:
                    timestamp_str = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp_str}] [{entry['component']}] [{entry['level']}] {entry['message']}")
                    if entry['details']:
                        f.write(f" - {entry['details']}")
                    f.write("\n")
            
            # Export statistics
            stats_file = os.path.join(export_dir, "log_statistics.json")
            with open(stats_file, 'w') as f:
                json.dump(self.log_statistics, f, indent=2)
            
        except Exception as e:
            self._add_log_entry("SYSTEM", "ERROR", f"Data export failed: {str(e)}")