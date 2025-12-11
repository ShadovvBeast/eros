"""
Built-in tool collections for the Autonomous Agent.

This package contains comprehensive built-in tools organized by category.
"""

from .file_operations import create_file_tools
from .web_operations import create_web_tools
from .data_processing import create_data_tools
from .security_tools import create_security_tools
from .math_tools import create_math_tools
from .system_tools import create_system_tools

__all__ = [
    'create_file_tools',
    'create_web_tools', 
    'create_data_tools',
    'create_security_tools',
    'create_math_tools',
    'create_system_tools',
    'get_all_builtin_tools'
]


def get_all_builtin_tools():
    """Get all built-in tools organized by category"""
    tools = []
    
    tools.extend(create_file_tools())
    tools.extend(create_web_tools())
    tools.extend(create_data_tools())
    tools.extend(create_security_tools())
    tools.extend(create_math_tools())
    tools.extend(create_system_tools())
    
    return tools