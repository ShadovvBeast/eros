"""
Dashboard Tab Components

Individual tab implementations for the dashboard.
"""

from .health_tab import HealthTab
from .hardware_tab import HardwareTab
from .pathos_tab import PathosTab
from .memory_tab import MemoryTab
from .preference_tab import PreferenceTab
from .attractor_tab import AttractorTab
from .performance_tab import PerformanceTab
from .thread_tab import ThreadManagementTab
from .log_tab import LogAuditTab

__all__ = [
    'HealthTab',
    'HardwareTab', 
    'PathosTab',
    'MemoryTab',
    'PreferenceTab',
    'AttractorTab',
    'PerformanceTab',
    'ThreadManagementTab',
    'LogAuditTab'
]