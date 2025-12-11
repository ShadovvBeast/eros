"""
Tests for Tool Layer implementation.
"""

import pytest
import time
import uuid
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import Dict, Any, List

from src.tools import ToolLayer, MCPTool
from src.models import ToolCall, ToolResult
from src.config import ToolConfig
from src.ethos.ethos_framework import ConcreteEthosFramework
from src.ethos.interfaces import EthosFramework


class TestMCPTool:
    """Test cases for MCPTool implementation"""
    
    def test_mcp_tool_creation(self):
        """Test basic MCPTool creation and properties"""
        def dummy_execute(args):
            return "test_result"
        
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            category="testing",
            execute_func=dummy_execute
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.category == "testing"
    
    def test_mcp_tool_execution_success(self):
        """Test successful tool execution"""
        def dummy_execute(args):
            return f"Processed: {args.get('input', 'default')}"
        
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            category="testing",
            execute_func=dummy_execute
        )
        
        result = tool.execute({"input": "test_data"})
        
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.result == "Processed: test_data"
        assert result.error_message is None
    
    def test_mcp_tool_execution_failure(self):
        """Test tool execution with exception"""
        def failing_execute(args):
            raise ValueError("Test error")
        
        tool = MCPTool(
            name="failing_tool",
            description="A failing tool",
            category="testing",
            execute_func=failing_execute
        )
        
        result = tool.execute({"input": "test_data"})
        
        assert result.success is False
        assert result.tool_name == "failing_tool"
        assert result.result is None
        assert "Test error" in result.error_message
    
    def test_mcp_tool_validation(self):
        """Test tool argument validation"""
        def dummy_execute(args):
            return "result"
        
        def custom_validate(args):
            if "required_field" not in args:
                return False, "Missing required_field"
            return True, None
        
        tool = MCPTool(
            name="validating_tool",
            description="A tool with validation",
            category="testing",
            execute_func=dummy_execute,
            validate_func=custom_validate
        )
        
        # Test valid arguments
        is_valid, error = tool.validate_arguments({"required_field": "value"})
        assert is_valid is True
        assert error is None
        
        # Test invalid arguments
        is_valid, error = tool.validate_arguments({"other_field": "value"})
        assert is_valid is False
        assert "Missing required_field" in error


class TestToolLayer:
    """Test cases for ToolLayer implementation"""
    
    @pytest.fixture
    def tool_config(self):
        """Create a test tool configuration"""
        return ToolConfig(
            max_concurrent_tools=2,
            tool_timeout_seconds=5,
            allowed_tool_types=['testing', 'communication', 'information_gathering', 'analysis']
        )
    
    @pytest.fixture
    def ethos_framework(self):
        """Create a mock ethos framework"""
        framework = Mock(spec=EthosFramework)
        framework.validate_tool_call.return_value = (True, None)
        return framework
    
    @pytest.fixture
    def tool_layer(self, tool_config, ethos_framework):
        """Create a ToolLayer instance for testing"""
        return ToolLayer(tool_config, ethos_framework)
    
    def test_tool_layer_initialization(self, tool_layer):
        """Test ToolLayer initialization"""
        assert tool_layer.config is not None
        assert tool_layer.ethos_framework is not None
        assert len(tool_layer.registered_tools) > 0  # Should have default tools
        
        # Check default tools are registered
        available_tools = tool_layer.get_available_tools()
        assert "echo" in available_tools
        assert "information_gather" in available_tools
        assert "analyze_data" in available_tools
    
    def test_tool_registration(self, tool_layer):
        """Test tool registration"""
        def dummy_execute(args):
            return "test_result"
        
        test_tool = MCPTool(
            name="custom_tool",
            description="A custom test tool",
            category="testing",
            execute_func=dummy_execute
        )
        
        tool_layer.register_tool(test_tool)
        
        assert "custom_tool" in tool_layer.registered_tools
        assert tool_layer.get_tool_description("custom_tool") == "A custom test tool"
    
    def test_tool_registration_invalid_category(self, tool_layer):
        """Test tool registration with invalid category"""
        def dummy_execute(args):
            return "test_result"
        
        test_tool = MCPTool(
            name="invalid_tool",
            description="A tool with invalid category",
            category="forbidden_category",
            execute_func=dummy_execute
        )
        
        with pytest.raises(ValueError, match="Tool category forbidden_category not allowed"):
            tool_layer.register_tool(test_tool)
    
    def test_tool_execution_success(self, tool_layer):
        """Test successful tool execution"""
        tool_call = ToolCall(
            tool_name="echo",
            arguments={"message": "Hello, World!"},
            timestamp=int(time.time())
        )
        
        result = tool_layer.execute_tool(tool_call)
        
        assert result.success is True
        assert result.tool_name == "echo"
        assert "Hello, World!" in str(result.result)
    
    def test_tool_execution_nonexistent_tool(self, tool_layer):
        """Test execution of non-existent tool"""
        tool_call = ToolCall(
            tool_name="nonexistent_tool",
            arguments={},
            timestamp=int(time.time())
        )
        
        result = tool_layer.execute_tool(tool_call)
        
        assert result.success is False
        assert "not registered" in result.error_message
    
    def test_tool_validation(self, tool_layer):
        """Test tool call validation"""
        # Valid tool call
        valid_call = ToolCall(
            tool_name="echo",
            arguments={"message": "test"},
            timestamp=int(time.time())
        )
        
        is_valid, error = tool_layer.validate_tool_call(valid_call)
        assert is_valid is True
        assert error is None
        
        # Invalid tool call (missing required argument)
        invalid_call = ToolCall(
            tool_name="echo",
            arguments={},  # Missing 'message' argument
            timestamp=int(time.time())
        )
        
        is_valid, error = tool_layer.validate_tool_call(invalid_call)
        assert is_valid is False
        assert "message" in error
    
    def test_ethos_validation_integration(self, tool_config):
        """Test integration with Ethos Framework validation"""
        # Create a mock ethos framework that rejects certain tools
        ethos_framework = Mock(spec=EthosFramework)
        ethos_framework.validate_tool_call.return_value = (False, "Tool violates safety constraints")
        
        tool_layer = ToolLayer(tool_config, ethos_framework)
        
        tool_call = ToolCall(
            tool_name="echo",
            arguments={"message": "test"},
            timestamp=int(time.time())
        )
        
        is_valid, error = tool_layer.validate_tool_call(tool_call)
        assert is_valid is False
        assert "Ethos validation failed" in error
        assert "safety constraints" in error
    
    def test_result_callback_registration(self, tool_layer):
        """Test result callback registration and distribution"""
        callback_results = []
        
        def test_callback(result: ToolResult):
            callback_results.append(result)
        
        tool_layer.register_result_callback(test_callback)
        
        tool_call = ToolCall(
            tool_name="echo",
            arguments={"message": "callback test"},
            timestamp=int(time.time())
        )
        
        result = tool_layer.execute_tool(tool_call)
        
        # Check that callback was called
        assert len(callback_results) == 1
        assert callback_results[0].tool_name == "echo"
        assert callback_results[0].success is True
    
    def test_emergency_termination(self, tool_layer):
        """Test emergency termination of active executions"""
        # This is a basic test - in practice, we'd need long-running tools to test properly
        tool_layer.emergency_terminate_all()
        
        # Should not raise any exceptions
        assert len(tool_layer.active_executions) == 0
    
    def test_get_tool_categories(self, tool_layer):
        """Test getting tools organized by category"""
        categories = tool_layer.get_tool_categories()
        
        assert isinstance(categories, dict)
        assert "communication" in categories
        assert "echo" in categories["communication"]
        assert "information_gathering" in categories
        assert "information_gather" in categories["information_gathering"]
    
    def test_safety_enforcement_emergency_stop(self, tool_layer):
        """Test emergency stop functionality"""
        # Normal execution should work
        tool_call = ToolCall(
            tool_name="echo",
            arguments={"message": "test"},
            timestamp=int(time.time())
        )
        
        result = tool_layer.execute_tool(tool_call)
        assert result.success is True
        
        # Activate emergency stop
        tool_layer.emergency_terminate_all()
        
        # Tool execution should be blocked
        result = tool_layer.execute_tool(tool_call)
        assert result.success is False
        assert "Emergency stop is active" in result.error_message
        
        # Reset emergency stop
        tool_layer.reset_emergency_stop()
        
        # Tool execution should work again
        result = tool_layer.execute_tool(tool_call)
        assert result.success is True
    
    def test_capability_extension_with_safety(self, tool_layer):
        """Test safe capability extension"""
        def safe_execute(args):
            return "safe_result"
        
        safe_tool = MCPTool(
            name="safe_extension",
            description="A safe tool extension",
            category="testing",
            execute_func=safe_execute
        )
        
        # Should succeed with safety check
        success = tool_layer.extend_tool_capabilities(safe_tool, safety_check=True)
        assert success is True
        assert "safe_extension" in tool_layer.registered_tools
        
        # Try to add unsafe tool (invalid category)
        unsafe_tool = MCPTool(
            name="unsafe_extension",
            description="An unsafe tool extension",
            category="forbidden",
            execute_func=safe_execute
        )
        
        success = tool_layer.extend_tool_capabilities(unsafe_tool, safety_check=True)
        assert success is False
        assert "unsafe_extension" not in tool_layer.registered_tools
    
    def test_safety_status_reporting(self, tool_layer):
        """Test safety status reporting"""
        status = tool_layer.get_safety_status()
        
        assert isinstance(status, dict)
        assert "emergency_stop_active" in status
        assert "total_safety_violations" in status
        assert "active_executions" in status
        assert "registered_tools" in status
        assert status["emergency_stop_active"] is False
        assert status["registered_tools"] > 0
    
    def test_tool_usage_recording(self, tool_layer):
        """Test tool usage recording for safety monitoring"""
        tool_call = ToolCall(
            tool_name="echo",
            arguments={"message": "usage_test"},
            timestamp=int(time.time())
        )
        
        # Execute tool multiple times
        for i in range(3):
            result = tool_layer.execute_tool(tool_call)
            assert result.success is True
        
        # Check usage was recorded
        assert "echo" in tool_layer._tool_usage_history
        assert len(tool_layer._tool_usage_history["echo"]) == 3


# Property-based tests
class TestToolLayerProperties:
    """Property-based tests for Tool Layer"""
    
    @pytest.fixture
    def tool_config(self):
        return ToolConfig(
            max_concurrent_tools=3,
            tool_timeout_seconds=10,
            allowed_tool_types=['testing', 'communication', 'information_gathering', 'analysis']
        )
    
    @pytest.fixture
    def ethos_framework(self):
        framework = Mock(spec=EthosFramework)
        framework.validate_tool_call.return_value = (True, None)
        return framework
    
    @pytest.fixture
    def tool_layer(self, tool_config, ethos_framework):
        return ToolLayer(tool_config, ethos_framework)
    
    @given(
        tool_name=st.sampled_from(['echo', 'information_gather', 'analyze_data']),
        message=st.text(min_size=1, max_size=100),
        timestamp=st.integers(min_value=0, max_value=2**31-1)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_tool_execution_and_result_distribution(self, tool_layer, tool_name, message, timestamp):
        """
        **Feature: autonomous-logos-pathos-agent, Property 9: Tool Execution and Result Distribution**
        **Validates: Requirements 4.1, 4.2, 4.3**
        
        For any tool call requested by Logos, the tool should be executed with provided arguments 
        and results should be delivered to both Pathos and Logos layers.
        """
        # Register callbacks to simulate Pathos and Logos layers
        pathos_results = []
        logos_results = []
        
        def pathos_callback(result: ToolResult):
            pathos_results.append(result)
        
        def logos_callback(result: ToolResult):
            logos_results.append(result)
        
        tool_layer.register_result_callback(pathos_callback)
        tool_layer.register_result_callback(logos_callback)
        
        # Create appropriate arguments based on tool
        if tool_name == 'echo':
            arguments = {"message": message}
        elif tool_name == 'information_gather':
            arguments = {"query": message}
        elif tool_name == 'analyze_data':
            arguments = {"data": {"input": message}, "type": "test"}
        else:
            arguments = {}
        
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            timestamp=timestamp
        )
        
        # Execute the tool
        result = tool_layer.execute_tool(tool_call)
        
        # Verify tool execution
        assert result.tool_name == tool_name
        assert isinstance(result.success, bool)
        assert result.timestamp > 0
        
        # Verify result distribution to both layers
        assert len(pathos_results) == 1
        assert len(logos_results) == 1
        assert pathos_results[0].tool_name == tool_name
        assert logos_results[0].tool_name == tool_name
        
        # Results should be identical
        assert pathos_results[0].success == logos_results[0].success
        assert pathos_results[0].tool_name == logos_results[0].tool_name
    
    @given(
        tool_name=st.sampled_from(['echo', 'information_gather', 'analyze_data']),
        message=st.text(min_size=1, max_size=100),
        emergency_stop=st.booleans(),
        timestamp=st.integers(min_value=0, max_value=2**31-1)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_safety_enforcement(self, tool_layer, tool_name, message, emergency_stop, timestamp):
        """
        **Feature: autonomous-logos-pathos-agent, Property 17: Safety Enforcement**
        **Validates: Requirements 4.4, 8.3**
        
        For any tool usage or capability extension, safety restrictions defined in the 
        Ethos Framework should be enforced.
        """
        # Create appropriate arguments based on tool
        if tool_name == 'echo':
            arguments = {"message": message}
        elif tool_name == 'information_gather':
            arguments = {"query": message}
        elif tool_name == 'analyze_data':
            arguments = {"data": {"input": message}, "type": "test"}
        else:
            arguments = {}
        
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            timestamp=timestamp
        )
        
        # Test emergency stop enforcement
        if emergency_stop:
            tool_layer.emergency_terminate_all()
            
            # Tool execution should be blocked
            result = tool_layer.execute_tool(tool_call)
            assert result.success is False
            assert "Emergency stop is active" in result.error_message
            
            # Safety validation should also fail
            is_safe, safety_msg = tool_layer.validate_tool_safety(tool_call)
            assert is_safe is False
            assert "Emergency stop is active" in safety_msg
            
            # Reset for further testing
            tool_layer.reset_emergency_stop()
        
        # Test normal safety validation
        is_safe, safety_msg = tool_layer.validate_tool_safety(tool_call)
        
        # For valid tools with valid arguments, safety should pass
        if tool_name in tool_layer.registered_tools:
            tool = tool_layer.registered_tools[tool_name]
            is_valid_args, _ = tool.validate_arguments(arguments)
            
            if is_valid_args:
                assert is_safe is True
                assert safety_msg is None
        
        # Test safety status reporting
        safety_status = tool_layer.get_safety_status()
        assert isinstance(safety_status, dict)
        assert "emergency_stop_active" in safety_status
        assert "total_safety_violations" in safety_status
        assert safety_status["emergency_stop_active"] == tool_layer._emergency_stop_flag
        
        # Test capability extension safety
        def dummy_execute(args):
            return "test_result"
        
        # Try to extend with allowed category
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        allowed_tool = MCPTool(
            name=f"test_tool_{unique_id}_{timestamp}",
            description="Test tool for safety validation",
            category="testing",  # This is in allowed_tool_types
            execute_func=dummy_execute
        )
        
        success = tool_layer.extend_tool_capabilities(allowed_tool, safety_check=True)
        assert success is True
        
        # Try to extend with disallowed category
        disallowed_tool = MCPTool(
            name=f"bad_tool_{unique_id}_{timestamp}",
            description="Test tool with bad category",
            category="forbidden_category",  # This is not in allowed_tool_types
            execute_func=dummy_execute
        )
        
        success = tool_layer.extend_tool_capabilities(disallowed_tool, safety_check=True)
        assert success is False