"""
Memory Table Tab

Displays memory traces in a comprehensive tabular format.
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import List, Optional
import json

from .base_tab import BaseTab


class MemoryTableTab(BaseTab):
    """Memory table display tab showing all memory traces in tabular format."""
    
    def __init__(self, notebook, memory_traces):
        """Initialize memory table tab."""
        self.memory_traces = memory_traces
        super().__init__(notebook, "Memory Table", "📊")
        self._create_memory_table()
    
    def _create_memory_table(self):
        """Create the memory table display."""
        # Create main container with scrollbars
        container = ttk.Frame(self.frame)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview with scrollbars
        tree_frame = ttk.Frame(container)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Define columns
        columns = (
            'Index', 'Timestamp', 'Salience', 'Reward', 'Intention', 
            'State', 'Action', 'Observation', 'Reflection', 'Category'
        )
        
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        # Configure column headings and widths
        column_widths = {
            'Index': 60,
            'Timestamp': 120,
            'Salience': 80,
            'Reward': 80,
            'Intention': 200,
            'State': 150,
            'Action': 150,
            'Observation': 200,
            'Reflection': 200,
            'Category': 100
        }
        
        for col in columns:
            self.tree.heading(col, text=col, anchor=tk.W)
            self.tree.column(col, width=column_widths.get(col, 100), minwidth=50)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Add control buttons
        self._create_controls(container)
        
        # Add details panel
        self._create_details_panel(container)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self._on_item_select)
    
    def _create_controls(self, parent):
        """Create control buttons."""
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Refresh button
        refresh_btn = ttk.Button(
            controls_frame, 
            text="🔄 Refresh", 
            command=self.update_display
        )
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Export button
        export_btn = ttk.Button(
            controls_frame, 
            text="💾 Export CSV", 
            command=self._export_csv
        )
        export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Filter controls
        ttk.Label(controls_frame, text="Filter:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(controls_frame, textvariable=self.filter_var, width=20)
        filter_entry.pack(side=tk.LEFT, padx=(0, 5))
        filter_entry.bind('<KeyRelease>', self._on_filter_change)
        
        # Category filter
        ttk.Label(controls_frame, text="Category:").pack(side=tk.LEFT, padx=(20, 5))
        
        self.category_var = tk.StringVar(value="All")
        category_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.category_var, 
            values=["All", "Exploration", "Analysis", "Learning", "Reflection", "Other"],
            width=12,
            state="readonly"
        )
        category_combo.pack(side=tk.LEFT, padx=(0, 5))
        category_combo.bind('<<ComboboxSelected>>', self._on_filter_change)
        
        # Statistics label
        self.stats_label = ttk.Label(controls_frame, text="Total: 0 traces")
        self.stats_label.pack(side=tk.RIGHT)
    
    def _create_details_panel(self, parent):
        """Create details panel for selected trace."""
        details_frame = ttk.LabelFrame(parent, text="Trace Details", padding="10")
        details_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(details_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.details_text = tk.Text(
            text_frame, 
            height=8, 
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        details_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initially show help text
        self.details_text.insert(tk.END, "Select a memory trace from the table above to view detailed information.")
        self.details_text.config(state=tk.DISABLED)
    
    def update_display(self):
        """Update memory table with current data."""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Check if we have memory traces
            if not self.memory_traces or len(self.memory_traces) == 0:
                self._show_no_data_message()
                return
            
            # Apply filters
            filtered_traces = self._apply_filters()
            
            # Populate table
            for i, trace in enumerate(filtered_traces):
                values = self._extract_trace_values(i, trace)
                item_id = self.tree.insert('', tk.END, values=values)
                
                # Color coding based on salience
                if hasattr(trace, 'salience'):
                    if trace.salience > 0.8:
                        self.tree.set(item_id, 'Salience', f"{trace.salience:.3f} ⭐")
                    elif trace.salience > 0.6:
                        self.tree.set(item_id, 'Salience', f"{trace.salience:.3f} ⚡")
            
            # Update statistics
            self._update_statistics(filtered_traces)
            
        except Exception as e:
            self._show_error_message(f"Error updating memory table: {str(e)}")
    
    def _apply_filters(self):
        """Apply current filters to memory traces."""
        filtered = list(self.memory_traces)
        
        # Text filter
        text_filter = self.filter_var.get().lower().strip()
        if text_filter:
            filtered = [
                trace for trace in filtered
                if self._trace_matches_text_filter(trace, text_filter)
            ]
        
        # Category filter
        category_filter = self.category_var.get()
        if category_filter != "All":
            filtered = [
                trace for trace in filtered
                if self._get_trace_category(trace) == category_filter
            ]
        
        return filtered
    
    def _trace_matches_text_filter(self, trace, text_filter):
        """Check if trace matches text filter."""
        searchable_fields = []
        
        # Add intention if available
        if hasattr(trace, 'metadata') and trace.metadata and 'intention' in trace.metadata:
            searchable_fields.append(trace.metadata['intention'].lower())
        
        # Add state if available
        if hasattr(trace, 'state') and trace.state:
            searchable_fields.append(str(trace.state).lower())
        
        # Add action if available
        if hasattr(trace, 'action') and trace.action:
            searchable_fields.append(str(trace.action).lower())
        
        # Add observation if available
        if hasattr(trace, 'observation') and trace.observation:
            searchable_fields.append(str(trace.observation).lower())
        
        # Add reflection if available
        if hasattr(trace, 'reflection') and trace.reflection:
            searchable_fields.append(str(trace.reflection).lower())
        
        return any(text_filter in field for field in searchable_fields)
    
    def _extract_trace_values(self, index, trace):
        """Extract values from trace for table display."""
        # Format timestamp
        timestamp = "N/A"
        if hasattr(trace, 'timestamp'):
            if isinstance(trace.timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(trace.timestamp).strftime("%H:%M:%S")
            else:
                timestamp = str(trace.timestamp)
        
        # Extract salience
        salience = getattr(trace, 'salience', 0.0)
        salience_str = f"{salience:.3f}" if isinstance(salience, (int, float)) else str(salience)
        
        # Extract reward
        reward = getattr(trace, 'reward', 0.0)
        reward_str = f"{reward:.3f}" if isinstance(reward, (int, float)) else str(reward)
        
        # Extract intention
        intention = "N/A"
        if hasattr(trace, 'metadata') and trace.metadata and 'intention' in trace.metadata:
            intention = trace.metadata['intention'][:50] + "..." if len(trace.metadata['intention']) > 50 else trace.metadata['intention']
        
        # Extract state, action, observation, reflection
        state = self._truncate_text(getattr(trace, 'state', 'N/A'), 30)
        action = self._truncate_text(getattr(trace, 'action', 'N/A'), 30)
        observation = self._truncate_text(getattr(trace, 'observation', 'N/A'), 40)
        reflection = self._truncate_text(getattr(trace, 'reflection', 'N/A'), 40)
        
        # Determine category
        category = self._get_trace_category(trace)
        
        return (
            str(index + 1),
            timestamp,
            salience_str,
            reward_str,
            intention,
            state,
            action,
            observation,
            reflection,
            category
        )
    
    def _get_trace_category(self, trace):
        """Determine trace category based on content."""
        if hasattr(trace, 'metadata') and trace.metadata and 'intention' in trace.metadata:
            intention = trace.metadata['intention'].lower()
            if 'explore' in intention:
                return 'Exploration'
            elif 'analyze' in intention:
                return 'Analysis'
            elif 'learn' in intention:
                return 'Learning'
            elif 'reflect' in intention:
                return 'Reflection'
        return 'Other'
    
    def _truncate_text(self, text, max_length):
        """Truncate text to maximum length."""
        if text is None:
            return "N/A"
        text_str = str(text)
        return text_str[:max_length] + "..." if len(text_str) > max_length else text_str
    
    def _on_filter_change(self, event=None):
        """Handle filter changes."""
        self.update_display()
    
    def _on_item_select(self, event):
        """Handle item selection in the table."""
        selection = self.tree.selection()
        if not selection:
            return
        
        # Get selected item
        item = selection[0]
        values = self.tree.item(item, 'values')
        
        if not values:
            return
        
        # Get trace index
        try:
            trace_index = int(values[0]) - 1
            filtered_traces = self._apply_filters()
            
            if 0 <= trace_index < len(filtered_traces):
                trace = filtered_traces[trace_index]
                self._show_trace_details(trace)
        except (ValueError, IndexError):
            pass
    
    def _show_trace_details(self, trace):
        """Show detailed information for selected trace."""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        details = []
        details.append("=== MEMORY TRACE DETAILS ===\n")
        
        # Basic information
        if hasattr(trace, 'timestamp'):
            if isinstance(trace.timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(trace.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = str(trace.timestamp)
            details.append(f"Timestamp: {timestamp}")
        
        if hasattr(trace, 'salience'):
            details.append(f"Salience: {trace.salience}")
        
        if hasattr(trace, 'reward'):
            details.append(f"Reward: {trace.reward}")
        
        details.append("")
        
        # Metadata
        if hasattr(trace, 'metadata') and trace.metadata:
            details.append("--- Metadata ---")
            for key, value in trace.metadata.items():
                details.append(f"{key}: {value}")
            details.append("")
        
        # State
        if hasattr(trace, 'state') and trace.state:
            details.append("--- State ---")
            details.append(str(trace.state))
            details.append("")
        
        # Action
        if hasattr(trace, 'action') and trace.action:
            details.append("--- Action ---")
            details.append(str(trace.action))
            details.append("")
        
        # Observation
        if hasattr(trace, 'observation') and trace.observation:
            details.append("--- Observation ---")
            details.append(str(trace.observation))
            details.append("")
        
        # Reflection
        if hasattr(trace, 'reflection') and trace.reflection:
            details.append("--- Reflection ---")
            details.append(str(trace.reflection))
            details.append("")
        
        # Additional attributes
        other_attrs = []
        for attr in dir(trace):
            if not attr.startswith('_') and attr not in ['timestamp', 'salience', 'reward', 'metadata', 'state', 'action', 'observation', 'reflection']:
                try:
                    value = getattr(trace, attr)
                    if not callable(value):
                        other_attrs.append(f"{attr}: {value}")
                except:
                    pass
        
        if other_attrs:
            details.append("--- Other Attributes ---")
            details.extend(other_attrs)
        
        self.details_text.insert(tk.END, "\n".join(details))
        self.details_text.config(state=tk.DISABLED)
    
    def _update_statistics(self, traces):
        """Update statistics display."""
        total = len(traces)
        high_salience = len([t for t in traces if hasattr(t, 'salience') and t.salience > 0.7])
        
        stats_text = f"Total: {total} traces"
        if high_salience > 0:
            stats_text += f" | High salience: {high_salience}"
        
        self.stats_label.config(text=stats_text)
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        # Insert a single row with message
        self.tree.insert('', tk.END, values=(
            "No data", "N/A", "N/A", "N/A", 
            "No memory traces available", "Start an agent session", 
            "to see memory formation", "and trace data", "N/A", "N/A"
        ))
        
        self.stats_label.config(text="Total: 0 traces")
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        self.tree.insert('', tk.END, values=(
            "Error", "N/A", "N/A", "N/A", 
            f"Error: {error_msg}", "Check logs", 
            "for details", "N/A", "N/A", "N/A"
        ))
        
        self.stats_label.config(text="Error loading data")
    
    def _export_csv(self):
        """Export table data to CSV."""
        try:
            from tkinter import filedialog
            import csv
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Memory Table"
            )
            
            if filename:
                filtered_traces = self._apply_filters()
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    writer.writerow([
                        'Index', 'Timestamp', 'Salience', 'Reward', 'Intention',
                        'State', 'Action', 'Observation', 'Reflection', 'Category'
                    ])
                    
                    # Write data
                    for i, trace in enumerate(filtered_traces):
                        values = self._extract_trace_values(i, trace)
                        writer.writerow(values)
                
                tk.messagebox.showinfo("Export Complete", f"Data exported to {filename}")
        
        except Exception as e:
            tk.messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def export_data(self, export_dir: str):
        """Export tab data to specified directory."""
        try:
            import csv
            import os
            
            filename = os.path.join(export_dir, "memory_table.csv")
            filtered_traces = self._apply_filters()
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    'Index', 'Timestamp', 'Salience', 'Reward', 'Intention',
                    'State', 'Action', 'Observation', 'Reflection', 'Category'
                ])
                
                # Write data
                for i, trace in enumerate(filtered_traces):
                    values = self._extract_trace_values(i, trace)
                    writer.writerow(values)
        
        except Exception as e:
            print(f"Error exporting memory table data: {e}")