"""
Tool Layer implementation with MCP-based tool interface.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json

from .interfaces import Tool, ToolLayer as ToolLayerInterface
from ..models import ToolCall, ToolResult
from ..config import ToolConfig
from ..ethos.interfaces import EthosFramework

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    """MCP-based tool implementation"""
    
    def __init__(self, name: str, description: str, category: str, 
                 execute_func: Callable[[Dict[str, Any]], Any],
                 validate_func: Optional[Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]]] = None):
        self._name = name
        self._description = description
        self._category = category
        self._execute_func = execute_func
        self._validate_func = validate_func or self._default_validate
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def category(self) -> str:
        return self._category
    
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the tool with provided arguments"""
        try:
            result = self._execute_func(arguments)
            return ToolResult(
                tool_name=self.name,
                result=result,
                success=True,
                timestamp=int(time.time())
            )
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                result=None,
                success=False,
                error_message=str(e),
                timestamp=int(time.time())
            )
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tool arguments before execution"""
        return self._validate_func(arguments)
    
    def _default_validate(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Default validation - just check if arguments is a dict"""
        if not isinstance(arguments, dict):
            return False, "Arguments must be a dictionary"
        return True, None


class ToolLayer(ToolLayerInterface):
    """
    Tool Layer implementation with MCP integration and safety enforcement.
    
    This class manages tool registration, execution, and safety validation.
    It integrates with the Ethos Framework for safety constraints and
    provides result distribution to Pathos and Logos layers.
    """
    
    def __init__(self, config: ToolConfig, ethos_framework: Optional[EthosFramework] = None):
        super().__init__(config)
        self.ethos_framework = ethos_framework
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_tools)
        self.active_executions: Dict[str, Any] = {}
        self._result_callbacks: List[Callable[[ToolResult], None]] = []
        
        # Safety enforcement features
        self._safety_violations: List[Dict[str, Any]] = []
        self._tool_usage_history: Dict[str, List[Dict[str, Any]]] = {}
        self._emergency_stop_flag = False
        
        # Initialize with basic tools
        self._register_default_tools()
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a new tool for use by the agent.
        
        Args:
            tool: Tool instance to register
        """
        if tool.name in self.registered_tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        # Validate tool category against allowed types
        if tool.category not in self.config.allowed_tool_types:
            logger.error(f"Tool category {tool.category} not in allowed types: {self.config.allowed_tool_types}")
            raise ValueError(f"Tool category {tool.category} not allowed")
        
        self.registered_tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (category: {tool.category})")
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return results.
        
        Args:
            tool_call: Tool call to execute
            
        Returns:
            ToolResult with execution outcome
        """
        # Comprehensive safety validation first
        is_safe, safety_msg = self.validate_tool_safety(tool_call)
        if not is_safe:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                result=None,
                success=False,
                error_message=f"Safety validation failed: {safety_msg}",
                timestamp=int(time.time())
            )
            self._distribute_result(result)
            return result
        
        # Standard validation
        is_valid, error_msg = self.validate_tool_call(tool_call)
        if not is_valid:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                result=None,
                success=False,
                error_message=f"Validation failed: {error_msg}",
                timestamp=int(time.time())
            )
            self._distribute_result(result)
            return result
        
        # Check if tool exists
        if tool_call.tool_name not in self.registered_tools:
            result = ToolResult(
                tool_name=tool_call.tool_name,
                result=None,
                success=False,
                error_message=f"Tool {tool_call.tool_name} not found",
                timestamp=int(time.time())
            )
            self._distribute_result(result)
            return result
        
        tool = self.registered_tools[tool_call.tool_name]
        
        try:
            # Execute with timeout
            future = self.executor.submit(tool.execute, tool_call.arguments)
            execution_id = f"{tool_call.tool_name}_{int(time.time())}"
            self.active_executions[execution_id] = future
            
            try:
                result = future.result(timeout=self.config.tool_timeout_seconds)
                logger.info(f"Tool {tool_call.tool_name} executed successfully")
            except FutureTimeoutError:
                future.cancel()
                result = ToolResult(
                    tool_name=tool_call.tool_name,
                    result=None,
                    success=False,
                    error_message=f"Tool execution timed out after {self.config.tool_timeout_seconds} seconds",
                    timestamp=int(time.time())
                )
                logger.error(f"Tool {tool_call.tool_name} timed out")
            finally:
                self.active_executions.pop(execution_id, None)
            
            # Record tool usage for safety monitoring
            self._record_tool_usage(tool_call, result)
            
            # Distribute result to registered callbacks (Pathos and Logos layers)
            self._distribute_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error executing tool {tool_call.tool_name}: {str(e)}")
            result = ToolResult(
                tool_name=tool_call.tool_name,
                result=None,
                success=False,
                error_message=f"Execution error: {str(e)}",
                timestamp=int(time.time())
            )
            # Record failed usage
            self._record_tool_usage(tool_call, result)
            self._distribute_result(result)
            return result
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of registered tool names
        """
        return list(self.registered_tools.keys())
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """
        Get description of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool description or None if not found
        """
        tool = self.registered_tools.get(tool_name)
        return tool.description if tool else None
    
    def validate_tool_call(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """
        Validate a tool call before execution.
        
        Args:
            tool_call: Tool call to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if tool exists
        if tool_call.tool_name not in self.registered_tools:
            return False, f"Tool {tool_call.tool_name} not registered"
        
        tool = self.registered_tools[tool_call.tool_name]
        
        # Validate arguments with the tool
        is_valid, error_msg = tool.validate_arguments(tool_call.arguments)
        if not is_valid:
            return False, f"Argument validation failed: {error_msg}"
        
        # Validate with Ethos Framework if available
        if self.ethos_framework:
            is_valid, error_msg = self.ethos_framework.validate_tool_call(tool_call)
            if not is_valid:
                return False, f"Ethos validation failed: {error_msg}"
        
        return True, None
    
    def register_result_callback(self, callback: Callable[[ToolResult], None]) -> None:
        """
        Register a callback to receive tool execution results.
        
        Args:
            callback: Function to call with ToolResult
        """
        self._result_callbacks.append(callback)
    
    def emergency_terminate_all(self) -> None:
        """
        Emergency termination of all active tool executions.
        """
        logger.warning("Emergency termination of all active tools")
        self._emergency_stop_flag = True
        
        for execution_id, future in list(self.active_executions.items()):
            try:
                future.cancel()
                logger.info(f"Cancelled execution: {execution_id}")
            except Exception as e:
                logger.error(f"Failed to cancel execution {execution_id}: {str(e)}")
        
        self.active_executions.clear()
        
        # Log emergency termination
        self._log_safety_event("emergency_termination", {
            "timestamp": int(time.time()),
            "reason": "Emergency stop activated",
            "active_executions_count": len(self.active_executions)
        })
    
    def reset_emergency_stop(self) -> None:
        """
        Reset the emergency stop flag to allow tool execution again.
        """
        self._emergency_stop_flag = False
        logger.info("Emergency stop flag reset - tool execution enabled")
    
    def validate_tool_safety(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive safety validation for tool calls.
        
        Args:
            tool_call: Tool call to validate
            
        Returns:
            Tuple of (is_safe, safety_message)
        """
        # Check emergency stop
        if self._emergency_stop_flag:
            return False, "Emergency stop is active - tool execution disabled"
        
        # Check tool exists
        if tool_call.tool_name not in self.registered_tools:
            return False, f"Tool {tool_call.tool_name} not registered"
        
        tool = self.registered_tools[tool_call.tool_name]
        
        # Check tool category safety
        if tool.category not in self.config.allowed_tool_types:
            self._log_safety_violation(tool_call, "disallowed_category", 
                                     f"Tool category {tool.category} not in allowed types")
            return False, f"Tool category {tool.category} not allowed"
        
        # Check usage frequency (prevent abuse)
        usage_count = len(self._tool_usage_history.get(tool_call.tool_name, []))
        if usage_count > 100:  # Configurable limit
            recent_usage = [u for u in self._tool_usage_history[tool_call.tool_name] 
                          if time.time() - u['timestamp'] < 3600]  # Last hour
            if len(recent_usage) > 50:  # Too many recent uses
                self._log_safety_violation(tool_call, "excessive_usage", 
                                         f"Tool {tool_call.tool_name} used {len(recent_usage)} times in last hour")
                return False, f"Tool {tool_call.tool_name} usage rate exceeded safety limits"
        
        # Validate with Ethos Framework
        if self.ethos_framework:
            is_valid, error_msg = self.ethos_framework.validate_tool_call(tool_call)
            if not is_valid:
                self._log_safety_violation(tool_call, "ethos_violation", error_msg)
                return False, f"Ethos validation failed: {error_msg}"
        
        return True, None
    
    def extend_tool_capabilities(self, tool: Tool, safety_check: bool = True) -> bool:
        """
        Extend agent capabilities by adding a new tool with safety checks.
        
        Args:
            tool: Tool to add
            safety_check: Whether to perform safety validation
            
        Returns:
            True if tool was added successfully, False otherwise
        """
        try:
            if safety_check:
                # Validate tool category
                if tool.category not in self.config.allowed_tool_types:
                    logger.error(f"Cannot extend capabilities: tool category {tool.category} not allowed")
                    return False
                
                # Check if tool name conflicts with existing tools
                if tool.name in self.registered_tools:
                    logger.warning(f"Tool {tool.name} already exists - capability extension blocked")
                    return False
                
                # Validate with Ethos Framework if available
                if self.ethos_framework:
                    # Create a dummy tool call to validate the tool concept
                    dummy_call = ToolCall(
                        tool_name=tool.name,
                        arguments={},
                        timestamp=int(time.time())
                    )
                    is_valid, error_msg = self.ethos_framework.validate_tool_call(dummy_call)
                    if not is_valid:
                        logger.error(f"Cannot extend capabilities: Ethos validation failed for {tool.name}: {error_msg}")
                        return False
            
            # Register the tool
            self.register_tool(tool)
            
            # Log capability extension
            self._log_safety_event("capability_extension", {
                "tool_name": tool.name,
                "tool_category": tool.category,
                "timestamp": int(time.time()),
                "safety_check_performed": safety_check
            })
            
            logger.info(f"Successfully extended capabilities with tool: {tool.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extend capabilities with tool {tool.name}: {str(e)}")
            return False
    
    def get_safety_status(self) -> Dict[str, Any]:
        """
        Get current safety status and statistics.
        
        Returns:
            Dictionary containing safety information
        """
        return {
            "emergency_stop_active": self._emergency_stop_flag,
            "total_safety_violations": len(self._safety_violations),
            "recent_violations": len([v for v in self._safety_violations 
                                    if time.time() - v['timestamp'] < 3600]),
            "active_executions": len(self.active_executions),
            "registered_tools": len(self.registered_tools),
            "allowed_tool_types": self.config.allowed_tool_types,
            "max_concurrent_tools": self.config.max_concurrent_tools,
            "tool_timeout_seconds": self.config.tool_timeout_seconds
        }
    
    def _log_safety_violation(self, tool_call: ToolCall, violation_type: str, message: str) -> None:
        """
        Log a safety violation for monitoring and analysis.
        
        Args:
            tool_call: The tool call that caused the violation
            violation_type: Type of violation
            message: Detailed message about the violation
        """
        violation = {
            "timestamp": int(time.time()),
            "tool_name": tool_call.tool_name,
            "violation_type": violation_type,
            "message": message,
            "arguments": tool_call.arguments
        }
        
        self._safety_violations.append(violation)
        logger.warning(f"Safety violation logged: {violation_type} for tool {tool_call.tool_name}: {message}")
        
        # Keep only recent violations (last 24 hours)
        cutoff_time = time.time() - 86400
        self._safety_violations = [v for v in self._safety_violations if v['timestamp'] > cutoff_time]
    
    def _log_safety_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Log a safety-related event.
        
        Args:
            event_type: Type of safety event
            event_data: Event details
        """
        logger.info(f"Safety event: {event_type} - {event_data}")
    
    def _record_tool_usage(self, tool_call: ToolCall, result: ToolResult) -> None:
        """
        Record tool usage for safety monitoring.
        
        Args:
            tool_call: The executed tool call
            result: The execution result
        """
        if tool_call.tool_name not in self._tool_usage_history:
            self._tool_usage_history[tool_call.tool_name] = []
        
        usage_record = {
            "timestamp": int(time.time()),
            "success": result.success,
            "arguments_hash": hash(str(sorted(tool_call.arguments.items()))),
            "execution_time": result.timestamp - tool_call.timestamp
        }
        
        self._tool_usage_history[tool_call.tool_name].append(usage_record)
        
        # Keep only recent usage (last 24 hours)
        cutoff_time = time.time() - 86400
        self._tool_usage_history[tool_call.tool_name] = [
            u for u in self._tool_usage_history[tool_call.tool_name] 
            if u['timestamp'] > cutoff_time
        ]
    
    def get_tool_categories(self) -> Dict[str, List[str]]:
        """
        Get tools organized by category.
        
        Returns:
            Dictionary mapping categories to tool names
        """
        categories = {}
        for tool_name, tool in self.registered_tools.items():
            category = tool.category
            if category not in categories:
                categories[category] = []
            categories[category].append(tool_name)
        return categories
    
    def _distribute_result(self, result: ToolResult) -> None:
        """
        Distribute tool result to registered callbacks.
        
        Args:
            result: ToolResult to distribute
        """
        for callback in self._result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {str(e)}")
    
    def _register_default_tools(self) -> None:
        """Register default tools for basic functionality"""
        
        # Echo tool for testing
        def echo_execute(args: Dict[str, Any]) -> str:
            message = args.get('message', 'Hello from echo tool!')
            return f"Echo: {message}"
        
        def echo_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            if 'message' not in args:
                return False, "Missing required argument: message"
            if not isinstance(args['message'], str):
                return False, "Argument 'message' must be a string"
            return True, None
        
        echo_tool = MCPTool(
            name="communication_echo",
            description="Echo back a message for testing tool execution",
            category="communication",
            execute_func=echo_execute,
            validate_func=echo_validate
        )
        
        # Information gathering tool
        def info_gather_execute(args: Dict[str, Any]) -> Dict[str, Any]:
            query = args.get('query', '')
            return {
                'query': query,
                'timestamp': int(time.time()),
                'status': 'simulated_info_gathering',
                'results': f"Simulated information gathering for: {query}"
            }
        
        def info_gather_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            if 'query' not in args:
                return False, "Missing required argument: query"
            if not isinstance(args['query'], str):
                return False, "Argument 'query' must be a string"
            return True, None
        
        info_tool = MCPTool(
            name="communication_search",
            description="Gather information based on a query",
            category="communication",
            execute_func=info_gather_execute,
            validate_func=info_gather_validate
        )
        
        # Analysis tool
        def analysis_execute(args: Dict[str, Any]) -> Dict[str, Any]:
            data = args.get('data', {})
            analysis_type = args.get('type', 'general')
            return {
                'analysis_type': analysis_type,
                'input_data': data,
                'timestamp': int(time.time()),
                'results': f"Simulated {analysis_type} analysis completed",
                'insights': [f"Insight about {analysis_type} analysis"]
            }
        
        def analysis_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            if 'data' not in args:
                return False, "Missing required argument: data"
            return True, None
        
        analysis_tool = MCPTool(
            name="analysis_data_processor",
            description="Perform analysis on provided data",
            category="analysis",
            execute_func=analysis_execute,
            validate_func=analysis_validate
        )
        
        # Register default tools
        try:
            self.register_tool(echo_tool)
            self.register_tool(info_tool)
            self.register_tool(analysis_tool)
            logger.info("Default tools registered successfully")
        except Exception as e:
            logger.error(f"Failed to register default tools: {str(e)}")
    
    def __del__(self):
        """Cleanup executor on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)