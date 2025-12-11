"""
Hardware Monitoring Tab

Comprehensive hardware monitoring and visualization.
"""

from .base_tab import BaseTab


class HardwareTab(BaseTab):
    """Hardware monitoring tab."""
    
    def __init__(self, notebook, hardware_monitor):
        """Initialize hardware tab."""
        self.hardware_monitor = hardware_monitor
        super().__init__(notebook, "Hardware Monitor", "🖥️")
        # Implementation will be added in next iteration
    
    def update_display(self):
        """Update hardware display."""
        pass