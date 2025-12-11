"""
MCP tool wrapper for integrating MCP tools with the agent's tool system.
"""

import time
from typing import Dict, Any, Optional, Tuple

from ..interfaces import Tool
from ...core.models import ToolResult
from .server_manager import MCPServerManager, MCPToolSchema


class MCPToolWrapper(Tool):
    """Wrapper for MCP server tools to integrate with the agent's tool system"""
    
    def __init__(self, schema: MCPToolSchema, server_manager: MCPServerManager):
        self.schema = schema
        self.server_manager = server_manager
    
    @property
    def name(self) -> str:
        return f"mcp.{self.schema.server_name}.{self.schema.name}"
    
    @property
    def description(self) -> str:
        return f"[MCP:{self.schema.server_name}] {self.schema.description}"
    
    @property
    def category(self) -> str:
        return "mcp_tools"
    
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the MCP tool"""
        try:
            result = self.server_manager.execute_tool(
                self.schema.server_name,
                self.schema.name,
                arguments
            )
            
            return ToolResult(
                tool_name=self.name,
                result=result,
                success=result.get("success", False),
                error_message=result.get("error") if not result.get("success", False) else None,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                result=None,
                success=False,
                error_message=str(e),
                timestamp=int(time.time())
            )
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate arguments against the tool's input schema"""
        try:
            # Basic validation against schema
            schema = self.schema.input_schema
            
            if "properties" in schema:
                required = schema.get("required", [])
                
                # Check required fields
                for field in required:
                    if field not in arguments:
                        return False, f"Missing required field: {field}"
                
                # Check field types (basic validation)
                for field, value in arguments.items():
                    if field in schema["properties"]:
                        expected_type = schema["properties"][field].get("type")
                        if expected_type:
                            if expected_type == "string" and not isinstance(value, str):
                                return False, f"Field {field} must be a string"
                            elif expected_type == "integer" and not isinstance(value, int):
                                return False, f"Field {field} must be an integer"
                            elif expected_type == "number" and not isinstance(value, (int, float)):
                                return False, f"Field {field} must be a number"
                            elif expected_type == "boolean" and not isinstance(value, bool):
                                return False, f"Field {field} must be a boolean"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"