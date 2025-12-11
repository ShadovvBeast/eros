"""
E.R.O.S — Enhanced Reasoning & Operational System

Main package for the autonomous agent system.
"""

# Core components
from .core import (
    AutonomousAgent,
    AgentConfig, PathosConfig, LogosConfig, MemoryConfig, EthosConfig, ToolConfig,
    PathosState, SemanticVector, MemoryTrace, Intention, ToolCall, ToolResult,
    InstrumentationCollector, AgentLogger
)

# Monitoring components  
from .monitoring import (
    SessionManager, SessionConfig, SessionState, get_session_manager,
    HardwareMonitor, get_hardware_monitor,
    ThreadManager, get_thread_manager, register_thread
)

# Layer implementations
from .logos.logos_layer import LogosLayer
from .pathos.pathos_layer import PathosLayer
from .memory.memory_system import ConcreteMemorySystem
from .ethos.ethos_framework import ConcreteEthosFramework
from .tools.tool_layer import ToolLayer

# Agent implementations
from .agents import SimpleAgent, UltimateAutonomousAgent

# Dashboard and visualization
from .dashboard import InteractiveDashboard
from .visualization import (
    PathosStateVisualizer, PreferenceDriftAnalyzer, AttractorPatternDetector,
    MemoryNetworkVisualizer, SystemHealthDashboard
)

__version__ = "1.0.0"

__all__ = [
    # Core
    'AutonomousAgent',
    'AgentConfig', 'PathosConfig', 'LogosConfig', 'MemoryConfig', 'EthosConfig', 'ToolConfig',
    'PathosState', 'SemanticVector', 'MemoryTrace', 'Intention', 'ToolCall', 'ToolResult',
    'InstrumentationCollector', 'AgentLogger',
    
    # Monitoring
    'SessionManager', 'SessionConfig', 'SessionState', 'get_session_manager',
    'HardwareMonitor', 'get_hardware_monitor',
    'ThreadManager', 'get_thread_manager', 'register_thread',
    
    # Layers
    'LogosLayer', 'PathosLayer', 'ConcreteMemorySystem', 'ConcreteEthosFramework', 'ToolLayer',
    
    # Agents
    'SimpleAgent', 'UltimateAutonomousAgent',
    
    # Dashboard & Visualization
    'InteractiveDashboard',
    'PathosStateVisualizer', 'PreferenceDriftAnalyzer', 'AttractorPatternDetector',
    'MemoryNetworkVisualizer', 'SystemHealthDashboard'
]