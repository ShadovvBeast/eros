"""
File operation tools for the autonomous agent.
"""

import os
import json
import csv
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..interfaces import Tool
from ...models import ToolResult


class FileOperationTool(Tool):
    """Base class for file operation tools"""
    
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
        return "file_operations"
    
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


def create_file_tools() -> List[Tool]:
    """Create file operation tools"""
    tools = []
    
    # Read file tool
    def read_file_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        file_path = args['path']
        encoding = args.get('encoding', 'utf-8')
        
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return {
            "success": True,
            "content": content,
            "file_size": len(content),
            "encoding": encoding,
            "path": file_path
        }
    
    def read_file_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if 'path' not in args:
            return False, "Missing required argument: path"
        if not isinstance(args['path'], str):
            return False, "Path must be a string"
        return True, None
    
    tools.append(FileOperationTool(
        "read_file",
        "Read content from a file",
        read_file_execute,
        read_file_validate
    ))
    
    # Write file tool
    def write_file_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        file_path = args['path']
        content = args['content']
        encoding = args.get('encoding', 'utf-8')
        mode = args.get('mode', 'w')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        
        return {
            "success": True,
            "bytes_written": len(content.encode(encoding)),
            "path": file_path,
            "mode": mode
        }
    
    def write_file_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if 'path' not in args:
            return False, "Missing required argument: path"
        if 'content' not in args:
            return False, "Missing required argument: content"
        return True, None
    
    tools.append(FileOperationTool(
        "write_file",
        "Write content to a file",
        write_file_execute,
        write_file_validate
    ))
    
    # List directory tool
    def list_directory_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        directory_path = args['path']
        recursive = args.get('recursive', False)
        include_hidden = args.get('include_hidden', False)
        
        files = []
        directories = []
        
        if recursive:
            for root, dirs, filenames in os.walk(directory_path):
                for filename in filenames:
                    if not include_hidden and filename.startswith('.'):
                        continue
                    full_path = os.path.join(root, filename)
                    stat_info = os.stat(full_path)
                    files.append({
                        "name": filename,
                        "path": full_path,
                        "size": stat_info.st_size,
                        "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    })
                
                for dirname in dirs:
                    if not include_hidden and dirname.startswith('.'):
                        continue
                    full_path = os.path.join(root, dirname)
                    directories.append({
                        "name": dirname,
                        "path": full_path,
                        "type": "directory"
                    })
        else:
            for item in os.listdir(directory_path):
                if not include_hidden and item.startswith('.'):
                    continue
                
                full_path = os.path.join(directory_path, item)
                stat_info = os.stat(full_path)
                
                if os.path.isfile(full_path):
                    files.append({
                        "name": item,
                        "path": full_path,
                        "size": stat_info.st_size,
                        "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    })
                else:
                    directories.append({
                        "name": item,
                        "path": full_path,
                        "type": "directory"
                    })
        
        return {
            "success": True,
            "files": files,
            "directories": directories,
            "total_files": len(files),
            "total_directories": len(directories)
        }
    
    tools.append(FileOperationTool(
        "list_directory",
        "List contents of a directory",
        list_directory_execute
    ))
    
    # Search files tool
    def search_files_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        search_path = args['path']
        pattern = args['pattern']
        content_search = args.get('content_search', False)
        case_sensitive = args.get('case_sensitive', False)
        
        matches = []
        
        for root, dirs, files in os.walk(search_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Filename pattern matching
                filename_match = False
                if case_sensitive:
                    filename_match = pattern in file
                else:
                    filename_match = pattern.lower() in file.lower()
                
                # Content search if requested
                content_match = False
                if content_search and filename_match:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if case_sensitive:
                                content_match = pattern in content
                            else:
                                content_match = pattern.lower() in content.lower()
                    except:
                        content_match = False
                
                if filename_match or content_match:
                    stat_info = os.stat(file_path)
                    matches.append({
                        "path": file_path,
                        "filename": file,
                        "size": stat_info.st_size,
                        "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        "filename_match": filename_match,
                        "content_match": content_match
                    })
        
        return {
            "success": True,
            "matches": matches,
            "total_matches": len(matches),
            "search_pattern": pattern,
            "search_path": search_path
        }
    
    tools.append(FileOperationTool(
        "search_files",
        "Search for files by name or content",
        search_files_execute
    ))
    
    # JSON file processor
    def json_file_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # read, write, update
        file_path = args['path']
        
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "success": True,
                "data": data,
                "file_path": file_path
            }
        
        elif operation == 'write':
            data = args['data']
            indent = args.get('indent', 2)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent)
            
            return {
                "success": True,
                "file_path": file_path,
                "data_written": True
            }
        
        elif operation == 'update':
            # Read existing, update, write back
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {}
            
            updates = args['updates']
            existing_data.update(updates)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2)
            
            return {
                "success": True,
                "file_path": file_path,
                "updated_data": existing_data
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(FileOperationTool(
        "json_file_processor",
        "Read, write, and update JSON files",
        json_file_execute
    ))
    
    return tools