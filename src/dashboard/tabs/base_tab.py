"""
Base Tab Component

Base class for dashboard tab components.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import Optional
import os


class BaseTab:
    """Base class for dashboard tabs."""
    
    def __init__(self, notebook, title: str, icon: str = ""):
        """Initialize base tab."""
        self.notebook = notebook
        self.title = title
        self.icon = icon
        self.frame = None
        self.canvas = None
        self.figure = None
        
        self._create_tab()
    
    def _create_tab(self):
        """Create the tab frame."""
        self.frame = ttk.Frame(self.notebook, padding="10")
        tab_title = f"{self.icon} {self.title}" if self.icon else self.title
        self.notebook.add(self.frame, text=tab_title)
    
    def _add_matplotlib_canvas(self, fig: Figure):
        """Add matplotlib canvas to the tab."""
        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
    def update_display(self):
        """Update the tab display. Override in subclasses."""
        pass
    
    def export_data(self, export_dir: str):
        """Export tab data. Override in subclasses."""
        if self.figure:
            filename = f"{self.title.lower().replace(' ', '_')}.png"
            filepath = os.path.join(export_dir, filename)
            self.figure.savefig(filepath, dpi=300, bbox_inches='tight')
    
    def refresh(self):
        """Refresh the tab display."""
        if self.canvas:
            self.canvas.draw()