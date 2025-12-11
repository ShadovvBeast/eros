"""
Tool Layer interfaces and base classes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..models import ToolCall, ToolResult
from ..config import ToolConfig


class Tool(ABC):
    """Abstract base class for individual tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for Logos planning"""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Tool category for safety validation"""
        pass
    
    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with provided arguments.
        
        Args:
            arguments: Tool-specific arguments
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    @abstractmethod
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate tool arguments before execution.
        
        Args:
            arguments: Arguments to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class ToolLayer(ABC):
    """Abstract base class for the Tool Layer environment interface"""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.registered_tools: Dict[str, Tool] = {}
    
    @abstractmethod
    def register_tool(self, tool: Tool) -> None:
        """
        Register a new tool for use by the agent.
        
        Args:
            tool: Tool instance to register
        """
        pass
    
    @abstractmethod
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return results.
        
        Args:
            tool_call: Tool call to execute
            
        Returns:
            ToolResult with execution outcome
        """
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of registered tool names
        """
        pass
    
    @abstractmethod
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """
        Get description of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool description or None if not found
        """
        pass
    
    @abstractmethod
    def validate_tool_call(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """
        Validate a tool call before execution.
        
        Args:
            tool_call: Tool call to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass