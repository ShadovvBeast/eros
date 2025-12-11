"""
System monitoring and operation tools for the autonomous agent.
"""

import os
import sys
import subprocess
import time
import shutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import resource module with fallback for Windows
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

from ..interfaces import Tool
from ...core.models import ToolResult


class SystemTool(Tool):
    """Base class for system tools"""
    
    def __init__(self, name: str, description: str, execute_func, validate_func=None):
        self._name = name
        self._description = description
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
        return "system_operations"
    
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        try:
            result = self._execute_func(arguments)
            return ToolResult(
                tool_name=self.name,
                result=result,
                success=True,
                timestamp=int(datetime.now().timestamp())
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                result=None,
                success=False,
                error_message=str(e),
                timestamp=int(datetime.now().timestamp())
            )
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        return self._validate_func(arguments)
    
    def _default_validate(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(arguments, dict):
            return False, "Arguments must be a dictionary"
        return True, None


def create_system_tools() -> List[Tool]:
    """Create system monitoring and operation tools"""
    tools = []
    
    # System information tool
    def system_info_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        info_type = args.get('type', 'general')
        
        result = {"success": True, "timestamp": datetime.now().isoformat()}
        
        if info_type in ['general', 'all']:
            result.update({
                "platform": sys.platform,
                "python_version": sys.version,
                "current_directory": os.getcwd(),
                "process_id": os.getpid(),
                "environment_variable_count": len(os.environ)
            })
        
        if info_type in ['disk', 'all']:
            # Disk usage information
            disk_usage = shutil.disk_usage('.')
            result["disk_usage"] = {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent_used": (disk_usage.used / disk_usage.total) * 100
            }
        
        if info_type in ['memory', 'all']:
            # Basic memory info (if available)
            if HAS_RESOURCE:
                memory_usage = resource.getrusage(resource.RUSAGE_SELF)
                result["memory_usage"] = {
                    "max_rss": memory_usage.ru_maxrss,
                    "user_time": memory_usage.ru_utime,
                    "system_time": memory_usage.ru_stime
                }
            else:
                result["memory_usage"] = {
                    "note": "Memory usage info not available on this platform",
                    "platform": sys.platform
                }
        
        return result
    
    tools.append(SystemTool(
        "system_info",
        "Get system information and diagnostics",
        system_info_execute
    ))
    
    # Environment variable tool
    def env_var_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # get, set, list
        
        if operation == 'get':
            var_name = args['name']
            value = os.environ.get(var_name)
            return {
                "success": True,
                "name": var_name,
                "value": value,
                "exists": value is not None
            }
        
        elif operation == 'list':
            pattern = args.get('pattern', '')
            env_vars = {}
            
            for key, value in os.environ.items():
                if not pattern or pattern.lower() in key.lower():
                    env_vars[key] = value
            
            return {
                "success": True,
                "environment_variables": env_vars,
                "count": len(env_vars),
                "pattern": pattern
            }
        
        elif operation == 'set':
            var_name = args['name']
            var_value = args['value']
            os.environ[var_name] = str(var_value)
            
            return {
                "success": True,
                "name": var_name,
                "value": var_value,
                "set": True
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(SystemTool(
        "environment_variables",
        "Get, set, and list environment variables",
        env_var_execute
    ))
    
    # Process execution tool (with safety restrictions)
    def execute_command_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        command = args['command']
        shell = args.get('shell', True)
        timeout = args.get('timeout', 30)
        capture_output = args.get('capture_output', True)
        
        # Safety check - only allow safe commands
        safe_commands = [
            'echo', 'ls', 'dir', 'pwd', 'whoami', 'date', 'time',
            'python', 'pip', 'node', 'npm', 'git', 'curl', 'wget'
        ]
        
        command_parts = command.split()
        if command_parts and command_parts[0] not in safe_commands:
            raise ValueError(f"Command '{command_parts[0]}' not in allowed safe commands")
        
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        
        return {
            "success": True,
            "return_code": result.returncode,
            "stdout": result.stdout if capture_output else None,
            "stderr": result.stderr if capture_output else None,
            "command": command
        }
    
    tools.append(SystemTool(
        "execute_safe_command",
        "Execute safe system commands with restrictions",
        execute_command_execute
    ))
    
    # File permissions tool
    def file_permissions_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        file_path = args['path']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat_info = os.stat(file_path)
        
        # Get file permissions
        permissions = {
            "owner_read": bool(stat_info.st_mode & 0o400),
            "owner_write": bool(stat_info.st_mode & 0o200),
            "owner_execute": bool(stat_info.st_mode & 0o100),
            "group_read": bool(stat_info.st_mode & 0o040),
            "group_write": bool(stat_info.st_mode & 0o020),
            "group_execute": bool(stat_info.st_mode & 0o010),
            "other_read": bool(stat_info.st_mode & 0o004),
            "other_write": bool(stat_info.st_mode & 0o002),
            "other_execute": bool(stat_info.st_mode & 0o001)
        }
        
        # Convert to octal representation
        octal_permissions = oct(stat_info.st_mode)[-3:]
        
        return {
            "success": True,
            "path": file_path,
            "permissions": permissions,
            "octal_permissions": octal_permissions,
            "is_file": os.path.isfile(file_path),
            "is_directory": os.path.isdir(file_path),
            "size": stat_info.st_size,
            "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "created_time": datetime.fromtimestamp(stat_info.st_ctime).isoformat()
        }
    
    tools.append(SystemTool(
        "file_permissions",
        "Get file permissions and metadata",
        file_permissions_execute
    ))
    
    # Path utilities tool
    def path_utilities_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # join, split, absolute, relative, exists
        
        if operation == 'join':
            parts = args['parts']
            joined_path = os.path.join(*parts)
            return {
                "success": True,
                "parts": parts,
                "joined_path": joined_path,
                "normalized": os.path.normpath(joined_path)
            }
        
        elif operation == 'split':
            path = args['path']
            directory, filename = os.path.split(path)
            name, extension = os.path.splitext(filename)
            
            return {
                "success": True,
                "original_path": path,
                "directory": directory,
                "filename": filename,
                "name": name,
                "extension": extension
            }
        
        elif operation == 'absolute':
            path = args['path']
            absolute_path = os.path.abspath(path)
            return {
                "success": True,
                "original_path": path,
                "absolute_path": absolute_path
            }
        
        elif operation == 'relative':
            path = args['path']
            start = args.get('start', os.getcwd())
            relative_path = os.path.relpath(path, start)
            return {
                "success": True,
                "original_path": path,
                "start_path": start,
                "relative_path": relative_path
            }
        
        elif operation == 'exists':
            path = args['path']
            return {
                "success": True,
                "path": path,
                "exists": os.path.exists(path),
                "is_file": os.path.isfile(path),
                "is_directory": os.path.isdir(path),
                "is_link": os.path.islink(path)
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(SystemTool(
        "path_utilities",
        "Path manipulation and information utilities",
        path_utilities_execute
    ))
    
    return tools