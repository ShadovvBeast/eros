"""
Comprehensive Tool Toolkit for the Autonomous Agent

This module provides a complete suite of tools covering various domains:
- System operations and file management
- Web scraping and API interactions
- Data processing and analysis
- Machine learning and AI operations
- Creative tools and content generation
- Communication and collaboration
- Development and debugging tools
- Security and monitoring tools

All tools are designed to work in a sandboxed environment with full access.
"""

import os
import sys
import json
import time
import random
import hashlib
import subprocess
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import sqlite3
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import base64
import re
import math
import statistics

# Import the tool interfaces
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from src.tools.interfaces import Tool
from src.models import ToolResult


@dataclass
class ToolMetadata:
    """Metadata for tool registration"""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    complexity: str = "medium"
    requires_network: bool = False
    requires_filesystem: bool = False
    safety_level: str = "safe"  # safe, caution, restricted


class ComprehensiveTool(Tool):
    """Base class for comprehensive tools with enhanced functionality"""
    
    def __init__(self, metadata: ToolMetadata, execute_func, validate_func=None):
        self.metadata = metadata
        self._execute_func = execute_func
        self._validate_func = validate_func or self._default_validate
        self._usage_count = 0
        self._last_used = None
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def description(self) -> str:
        return self.metadata.description
    
    @property
    def category(self) -> str:
        return self.metadata.category
    
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute tool with enhanced logging and error handling"""
        self._usage_count += 1
        self._last_used = datetime.now()
        
        try:
            result = self._execute_func(arguments)
            return ToolResult(
                tool_name=self.name,
                result=result,
                success=True,
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
        return self._validate_func(arguments)
    
    def _default_validate(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(arguments, dict):
            return False, "Arguments must be a dictionary"
        return True, None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool"""
        return {
            "usage_count": self._usage_count,
            "last_used": self._last_used.isoformat() if self._last_used else None,
            "metadata": {
                "category": self.category,
                "complexity": self.metadata.complexity,
                "safety_level": self.metadata.safety_level
            }
        }


# ============================================================================
# SYSTEM AND FILE OPERATIONS
# ============================================================================

def create_file_system_tools() -> List[ComprehensiveTool]:
    """Create comprehensive file system operation tools"""
    tools = []
    
    # File operations
    def read_file_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        file_path = args['path']
        encoding = args.get('encoding', 'utf-8')
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "file_size": len(content),
                "encoding": encoding,
                "path": file_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="read_file",
            description="Read content from a file",
            category="file_operations",
            parameters={
                "path": {"type": "string", "description": "File path to read"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"}
            },
            requires_filesystem=True
        ),
        read_file_execute
    ))
    
    # Write file
    def write_file_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        file_path = args['path']
        content = args['content']
        encoding = args.get('encoding', 'utf-8')
        mode = args.get('mode', 'w')  # 'w' for write, 'a' for append
        
        try:
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
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="write_file",
            description="Write content to a file",
            category="file_operations",
            parameters={
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                "mode": {"type": "string", "description": "Write mode (w/a)", "default": "w"}
            },
            requires_filesystem=True
        ),
        write_file_execute
    ))
    
    return tools
    # Directory operations
    def list_directory_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        directory_path = args['path']
        recursive = args.get('recursive', False)
        include_hidden = args.get('include_hidden', False)
        
        try:
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
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="list_directory",
            description="List contents of a directory",
            category="file_operations",
            parameters={
                "path": {"type": "string", "description": "Directory path to list"},
                "recursive": {"type": "boolean", "description": "List recursively", "default": False},
                "include_hidden": {"type": "boolean", "description": "Include hidden files", "default": False}
            },
            requires_filesystem=True
        ),
        list_directory_execute
    ))
    
    # File search
    def search_files_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        search_path = args['path']
        pattern = args['pattern']
        content_search = args.get('content_search', False)
        case_sensitive = args.get('case_sensitive', False)
        
        try:
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
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="search_files",
            description="Search for files by name or content",
            category="file_operations",
            parameters={
                "path": {"type": "string", "description": "Directory to search in"},
                "pattern": {"type": "string", "description": "Search pattern"},
                "content_search": {"type": "boolean", "description": "Search file contents", "default": False},
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False}
            },
            requires_filesystem=True
        ),
        search_files_execute
    ))
    
    return tools


# ============================================================================
# WEB AND NETWORK OPERATIONS
# ============================================================================

def create_web_tools() -> List[ComprehensiveTool]:
    """Create web scraping and network operation tools"""
    tools = []
    
    # HTTP request tool
    def http_request_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        url = args['url']
        method = args.get('method', 'GET').upper()
        headers = args.get('headers', {})
        data = args.get('data', None)
        timeout = args.get('timeout', 30)
        
        try:
            # Prepare request
            if data and isinstance(data, dict):
                data = json.dumps(data).encode('utf-8')
                headers['Content-Type'] = 'application/json'
            elif data and isinstance(data, str):
                data = data.encode('utf-8')
            
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            
            # Make request
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode('utf-8')
                
                return {
                    "success": True,
                    "status_code": response.getcode(),
                    "headers": dict(response.headers),
                    "content": content,
                    "url": response.geturl(),
                    "content_length": len(content)
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="http_request",
            description="Make HTTP requests to web services",
            category="web_operations",
            parameters={
                "url": {"type": "string", "description": "URL to request"},
                "method": {"type": "string", "description": "HTTP method", "default": "GET"},
                "headers": {"type": "object", "description": "HTTP headers", "default": {}},
                "data": {"type": "string", "description": "Request body data"},
                "timeout": {"type": "integer", "description": "Request timeout", "default": 30}
            },
            requires_network=True
        ),
        http_request_execute
    ))
    
    # Web scraper
    def web_scrape_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        url = args['url']
        extract_links = args.get('extract_links', False)
        extract_images = args.get('extract_images', False)
        extract_text = args.get('extract_text', True)
        
        try:
            # Fetch the page
            with urllib.request.urlopen(url) as response:
                html_content = response.read().decode('utf-8')
            
            result = {
                "success": True,
                "url": url,
                "html_length": len(html_content)
            }
            
            if extract_text:
                # Simple text extraction (remove HTML tags)
                import re
                text = re.sub(r'<[^>]+>', '', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
                result["text_content"] = text
                result["text_length"] = len(text)
            
            if extract_links:
                # Extract links
                link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>'
                links = re.findall(link_pattern, html_content, re.IGNORECASE)
                result["links"] = [{"url": link[0], "text": link[1]} for link in links]
            
            if extract_images:
                # Extract images
                img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
                images = re.findall(img_pattern, html_content, re.IGNORECASE)
                result["images"] = images
            
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="web_scrape",
            description="Scrape content from web pages",
            category="web_operations",
            parameters={
                "url": {"type": "string", "description": "URL to scrape"},
                "extract_links": {"type": "boolean", "description": "Extract links", "default": False},
                "extract_images": {"type": "boolean", "description": "Extract images", "default": False},
                "extract_text": {"type": "boolean", "description": "Extract text content", "default": True}
            },
            requires_network=True
        ),
        web_scrape_execute
    ))
    
    return tools
# ============================================================================
# DATA PROCESSING AND ANALYSIS
# ============================================================================

def create_data_processing_tools() -> List[ComprehensiveTool]:
    """Create comprehensive data processing and analysis tools"""
    tools = []
    
    # JSON processor
    def json_process_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # parse, stringify, query, transform
        data = args['data']
        
        try:
            if operation == 'parse':
                if isinstance(data, str):
                    parsed = json.loads(data)
                    return {
                        "success": True,
                        "result": parsed,
                        "type": type(parsed).__name__,
                        "size": len(str(parsed))
                    }
                else:
                    return {"success": False, "error": "Data must be a string for parsing"}
            
            elif operation == 'stringify':
                result = json.dumps(data, indent=args.get('indent', None))
                return {
                    "success": True,
                    "result": result,
                    "size": len(result)
                }
            
            elif operation == 'query':
                # Simple JSON path querying
                path = args.get('path', '')
                current = data
                
                if path:
                    for key in path.split('.'):
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        elif isinstance(current, list) and key.isdigit():
                            current = current[int(key)]
                        else:
                            return {"success": False, "error": f"Path '{path}' not found"}
                
                return {
                    "success": True,
                    "result": current,
                    "path": path
                }
            
            elif operation == 'transform':
                # Apply transformations
                transformations = args.get('transformations', [])
                result = data
                
                for transform in transformations:
                    if transform['type'] == 'filter_keys':
                        if isinstance(result, dict):
                            keys = transform['keys']
                            result = {k: v for k, v in result.items() if k in keys}
                    elif transform['type'] == 'map_values':
                        if isinstance(result, dict):
                            func = transform['function']  # Simple string operations
                            if func == 'upper':
                                result = {k: str(v).upper() if isinstance(v, str) else v 
                                        for k, v in result.items()}
                
                return {
                    "success": True,
                    "result": result,
                    "transformations_applied": len(transformations)
                }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="json_processor",
            description="Process and manipulate JSON data",
            category="data_processing",
            parameters={
                "operation": {"type": "string", "description": "Operation to perform"},
                "data": {"type": "any", "description": "Data to process"},
                "path": {"type": "string", "description": "JSON path for querying"},
                "transformations": {"type": "array", "description": "List of transformations"}
            }
        ),
        json_process_execute
    ))
    
    # CSV processor
    def csv_process_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # read, write, analyze
        
        try:
            if operation == 'read':
                file_path = args['file_path']
                delimiter = args.get('delimiter', ',')
                
                with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile, delimiter=delimiter)
                    rows = list(reader)
                
                return {
                    "success": True,
                    "data": rows,
                    "row_count": len(rows),
                    "columns": reader.fieldnames if rows else []
                }
            
            elif operation == 'write':
                file_path = args['file_path']
                data = args['data']
                delimiter = args.get('delimiter', ',')
                
                if not data:
                    return {"success": False, "error": "No data provided"}
                
                fieldnames = data[0].keys() if isinstance(data[0], dict) else None
                
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    if fieldnames:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=delimiter)
                        writer.writeheader()
                        writer.writerows(data)
                    else:
                        writer = csv.writer(csvfile, delimiter=delimiter)
                        writer.writerows(data)
                
                return {
                    "success": True,
                    "rows_written": len(data),
                    "file_path": file_path
                }
            
            elif operation == 'analyze':
                data = args['data']
                
                if not data:
                    return {"success": False, "error": "No data provided"}
                
                analysis = {
                    "row_count": len(data),
                    "column_count": len(data[0]) if data else 0,
                    "columns": list(data[0].keys()) if isinstance(data[0], dict) else [],
                    "sample_rows": data[:5]  # First 5 rows as sample
                }
                
                # Basic statistics for numeric columns
                if isinstance(data[0], dict):
                    numeric_stats = {}
                    for col in analysis["columns"]:
                        values = []
                        for row in data:
                            try:
                                val = float(row[col])
                                values.append(val)
                            except (ValueError, TypeError):
                                continue
                        
                        if values:
                            numeric_stats[col] = {
                                "count": len(values),
                                "mean": statistics.mean(values),
                                "median": statistics.median(values),
                                "min": min(values),
                                "max": max(values),
                                "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                            }
                    
                    analysis["numeric_statistics"] = numeric_stats
                
                return {
                    "success": True,
                    "analysis": analysis
                }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="csv_processor",
            description="Process and analyze CSV data",
            category="data_processing",
            parameters={
                "operation": {"type": "string", "description": "Operation to perform"},
                "file_path": {"type": "string", "description": "CSV file path"},
                "data": {"type": "array", "description": "CSV data as array"},
                "delimiter": {"type": "string", "description": "CSV delimiter", "default": ","}
            },
            requires_filesystem=True
        ),
        csv_process_execute
    ))
    
    return tools


# ============================================================================
# SYSTEM MONITORING AND DIAGNOSTICS
# ============================================================================

def create_system_tools() -> List[ComprehensiveTool]:
    """Create system monitoring and diagnostic tools"""
    tools = []
    
    # System information
    def system_info_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        info_type = args.get('type', 'general')
        
        try:
            result = {"success": True, "timestamp": datetime.now().isoformat()}
            
            if info_type in ['general', 'all']:
                result.update({
                    "platform": sys.platform,
                    "python_version": sys.version,
                    "current_directory": os.getcwd(),
                    "environment_variables": dict(os.environ) if info_type == 'all' else len(os.environ),
                    "process_id": os.getpid()
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
                # Basic memory info (limited without psutil)
                import resource
                memory_usage = resource.getrusage(resource.RUSAGE_SELF)
                result["memory_usage"] = {
                    "max_rss": memory_usage.ru_maxrss,
                    "user_time": memory_usage.ru_utime,
                    "system_time": memory_usage.ru_stime
                }
            
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="system_info",
            description="Get system information and diagnostics",
            category="system_monitoring",
            parameters={
                "type": {"type": "string", "description": "Info type (general/disk/memory/all)", "default": "general"}
            }
        ),
        system_info_execute
    ))
    
    # Process execution
    def execute_command_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        command = args['command']
        shell = args.get('shell', True)
        timeout = args.get('timeout', 30)
        capture_output = args.get('capture_output', True)
        
        try:
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
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="execute_command",
            description="Execute system commands",
            category="system_operations",
            parameters={
                "command": {"type": "string", "description": "Command to execute"},
                "shell": {"type": "boolean", "description": "Use shell", "default": True},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                "capture_output": {"type": "boolean", "description": "Capture output", "default": True}
            },
            safety_level="caution"
        ),
        execute_command_execute
    ))
    
    return tools
# ============================================================================
# CREATIVE AND CONTENT GENERATION TOOLS
# ============================================================================

def create_creative_tools() -> List[ComprehensiveTool]:
    """Create creative and content generation tools"""
    tools = []
    
    # Text generator
    def text_generator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        generation_type = args['type']  # story, poem, article, summary
        prompt = args.get('prompt', '')
        length = args.get('length', 'medium')  # short, medium, long
        style = args.get('style', 'neutral')
        
        try:
            # Template-based text generation (mock AI)
            templates = {
                'story': [
                    "Once upon a time, {prompt}. The adventure began when...",
                    "In a world where {prompt}, our hero discovered...",
                    "The mysterious {prompt} led to an unexpected journey..."
                ],
                'poem': [
                    "Roses are red, violets are blue, {prompt} is true...",
                    "In the quiet of the night, {prompt} shines bright...",
                    "Dancing through time, {prompt} in rhyme..."
                ],
                'article': [
                    "Recent developments in {prompt} have shown significant progress...",
                    "Understanding {prompt} requires careful analysis...",
                    "The implications of {prompt} extend far beyond..."
                ],
                'summary': [
                    "In summary, {prompt} can be understood as...",
                    "The key points regarding {prompt} include...",
                    "To conclude, {prompt} represents..."
                ]
            }
            
            if generation_type not in templates:
                return {"success": False, "error": f"Unknown generation type: {generation_type}"}
            
            # Select template and generate content
            template = random.choice(templates[generation_type])
            base_content = template.format(prompt=prompt)
            
            # Extend based on length
            length_multipliers = {'short': 1, 'medium': 3, 'long': 6}
            multiplier = length_multipliers.get(length, 3)
            
            # Generate additional content
            extensions = [
                "This development opens new possibilities for innovation.",
                "The implications are far-reaching and significant.",
                "Further research in this area could yield valuable insights.",
                "The potential applications are numerous and exciting.",
                "This represents a breakthrough in our understanding."
            ]
            
            full_content = base_content
            for i in range(multiplier - 1):
                full_content += " " + random.choice(extensions)
            
            return {
                "success": True,
                "generated_text": full_content,
                "type": generation_type,
                "length": length,
                "style": style,
                "word_count": len(full_content.split()),
                "character_count": len(full_content)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="text_generator",
            description="Generate creative text content",
            category="creative_tools",
            parameters={
                "type": {"type": "string", "description": "Generation type (story/poem/article/summary)"},
                "prompt": {"type": "string", "description": "Generation prompt"},
                "length": {"type": "string", "description": "Content length", "default": "medium"},
                "style": {"type": "string", "description": "Writing style", "default": "neutral"}
            }
        ),
        text_generator_execute
    ))
    
    # Image metadata processor
    def image_processor_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # info, resize, convert
        image_path = args.get('image_path', '')
        
        try:
            if operation == 'info':
                # Get basic image info without external libraries
                stat_info = os.stat(image_path)
                
                # Try to determine image type from extension
                ext = os.path.splitext(image_path)[1].lower()
                image_types = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', 
                             '.gif': 'GIF', '.bmp': 'BMP', '.tiff': 'TIFF'}
                
                return {
                    "success": True,
                    "file_path": image_path,
                    "file_size": stat_info.st_size,
                    "file_type": image_types.get(ext, 'Unknown'),
                    "extension": ext,
                    "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                }
            
            elif operation == 'generate_placeholder':
                # Generate a simple placeholder image description
                width = args.get('width', 800)
                height = args.get('height', 600)
                color = args.get('color', 'blue')
                
                # Create a simple SVG placeholder
                svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="{color}"/>
  <text x="50%" y="50%" font-family="Arial" font-size="24" fill="white" 
        text-anchor="middle" dominant-baseline="middle">
    {width}x{height} Placeholder
  </text>
</svg>'''
                
                output_path = args.get('output_path', 'placeholder.svg')
                with open(output_path, 'w') as f:
                    f.write(svg_content)
                
                return {
                    "success": True,
                    "generated_image": output_path,
                    "width": width,
                    "height": height,
                    "format": "SVG"
                }
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="image_processor",
            description="Process and generate image content",
            category="creative_tools",
            parameters={
                "operation": {"type": "string", "description": "Operation to perform"},
                "image_path": {"type": "string", "description": "Path to image file"},
                "width": {"type": "integer", "description": "Image width"},
                "height": {"type": "integer", "description": "Image height"},
                "color": {"type": "string", "description": "Color for generation"}
            },
            requires_filesystem=True
        ),
        image_processor_execute
    ))
    
    return tools


# ============================================================================
# COMMUNICATION AND COLLABORATION TOOLS
# ============================================================================

def create_communication_tools() -> List[ComprehensiveTool]:
    """Create communication and collaboration tools"""
    tools = []
    
    # Message formatter
    def message_formatter_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        message = args['message']
        format_type = args.get('format', 'plain')  # plain, markdown, html, json
        
        try:
            if format_type == 'markdown':
                # Convert to markdown format
                formatted = f"## Message\n\n{message}\n\n*Generated at {datetime.now().isoformat()}*"
            
            elif format_type == 'html':
                # Convert to HTML format
                formatted = f"""
                <div class="message">
                    <h2>Message</h2>
                    <p>{message}</p>
                    <small>Generated at {datetime.now().isoformat()}</small>
                </div>
                """
            
            elif format_type == 'json':
                # Convert to JSON format
                formatted = json.dumps({
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "format": "json"
                }, indent=2)
            
            else:  # plain
                formatted = f"Message: {message}\nTimestamp: {datetime.now().isoformat()}"
            
            return {
                "success": True,
                "original_message": message,
                "formatted_message": formatted,
                "format": format_type,
                "length": len(formatted)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="message_formatter",
            description="Format messages for different communication channels",
            category="communication",
            parameters={
                "message": {"type": "string", "description": "Message to format"},
                "format": {"type": "string", "description": "Output format", "default": "plain"}
            }
        ),
        message_formatter_execute
    ))
    
    # Email composer (mock)
    def email_composer_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        to_address = args['to']
        subject = args['subject']
        body = args['body']
        email_type = args.get('type', 'plain')  # plain, formal, friendly
        
        try:
            # Template-based email composition
            templates = {
                'formal': {
                    'greeting': 'Dear Sir/Madam,',
                    'closing': 'Sincerely,\nAutonomous Agent'
                },
                'friendly': {
                    'greeting': 'Hi there!',
                    'closing': 'Best regards,\nYour AI Assistant'
                },
                'plain': {
                    'greeting': 'Hello,',
                    'closing': 'Best,\nAgent'
                }
            }
            
            template = templates.get(email_type, templates['plain'])
            
            formatted_email = f"""To: {to_address}
Subject: {subject}

{template['greeting']}

{body}

{template['closing']}
"""
            
            return {
                "success": True,
                "formatted_email": formatted_email,
                "to": to_address,
                "subject": subject,
                "type": email_type,
                "character_count": len(formatted_email)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="email_composer",
            description="Compose formatted emails",
            category="communication",
            parameters={
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body content"},
                "type": {"type": "string", "description": "Email type", "default": "plain"}
            }
        ),
        email_composer_execute
    ))
    
    return tools
# ============================================================================
# SECURITY AND ENCRYPTION TOOLS
# ============================================================================

def create_security_tools() -> List[ComprehensiveTool]:
    """Create security and encryption tools"""
    tools = []
    
    # Hash generator
    def hash_generator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        data = args['data']
        algorithm = args.get('algorithm', 'sha256')  # md5, sha1, sha256, sha512
        
        try:
            # Convert data to bytes if it's a string
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Generate hash
            if algorithm == 'md5':
                hash_obj = hashlib.md5(data_bytes)
            elif algorithm == 'sha1':
                hash_obj = hashlib.sha1(data_bytes)
            elif algorithm == 'sha256':
                hash_obj = hashlib.sha256(data_bytes)
            elif algorithm == 'sha512':
                hash_obj = hashlib.sha512(data_bytes)
            else:
                return {"success": False, "error": f"Unsupported algorithm: {algorithm}"}
            
            hex_hash = hash_obj.hexdigest()
            
            return {
                "success": True,
                "original_data": str(data)[:100] + "..." if len(str(data)) > 100 else str(data),
                "hash": hex_hash,
                "algorithm": algorithm,
                "hash_length": len(hex_hash)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="hash_generator",
            description="Generate cryptographic hashes",
            category="security",
            parameters={
                "data": {"type": "string", "description": "Data to hash"},
                "algorithm": {"type": "string", "description": "Hash algorithm", "default": "sha256"}
            }
        ),
        hash_generator_execute
    ))
    
    # Base64 encoder/decoder
    def base64_processor_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # encode, decode
        data = args['data']
        
        try:
            if operation == 'encode':
                if isinstance(data, str):
                    encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
                else:
                    encoded = base64.b64encode(str(data).encode('utf-8')).decode('utf-8')
                
                return {
                    "success": True,
                    "operation": "encode",
                    "original_data": str(data),
                    "encoded_data": encoded,
                    "original_length": len(str(data)),
                    "encoded_length": len(encoded)
                }
            
            elif operation == 'decode':
                try:
                    decoded_bytes = base64.b64decode(data)
                    decoded = decoded_bytes.decode('utf-8')
                    
                    return {
                        "success": True,
                        "operation": "decode",
                        "encoded_data": data,
                        "decoded_data": decoded,
                        "encoded_length": len(data),
                        "decoded_length": len(decoded)
                    }
                except Exception as decode_error:
                    return {"success": False, "error": f"Invalid base64 data: {str(decode_error)}"}
            
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="base64_processor",
            description="Encode and decode base64 data",
            category="security",
            parameters={
                "operation": {"type": "string", "description": "Operation (encode/decode)"},
                "data": {"type": "string", "description": "Data to process"}
            }
        ),
        base64_processor_execute
    ))
    
    # Password generator
    def password_generator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        length = args.get('length', 12)
        include_uppercase = args.get('include_uppercase', True)
        include_lowercase = args.get('include_lowercase', True)
        include_numbers = args.get('include_numbers', True)
        include_symbols = args.get('include_symbols', False)
        
        try:
            import string
            
            characters = ""
            if include_lowercase:
                characters += string.ascii_lowercase
            if include_uppercase:
                characters += string.ascii_uppercase
            if include_numbers:
                characters += string.digits
            if include_symbols:
                characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"
            
            if not characters:
                return {"success": False, "error": "No character types selected"}
            
            password = ''.join(random.choice(characters) for _ in range(length))
            
            # Calculate strength
            strength_score = 0
            if include_lowercase: strength_score += 1
            if include_uppercase: strength_score += 1
            if include_numbers: strength_score += 1
            if include_symbols: strength_score += 2
            if length >= 12: strength_score += 2
            if length >= 16: strength_score += 1
            
            strength_levels = {
                (0, 2): "Weak",
                (3, 4): "Medium", 
                (5, 6): "Strong",
                (7, 8): "Very Strong"
            }
            
            strength = "Weak"
            for (min_score, max_score), level in strength_levels.items():
                if min_score <= strength_score <= max_score:
                    strength = level
                    break
            
            return {
                "success": True,
                "password": password,
                "length": length,
                "strength": strength,
                "strength_score": strength_score,
                "character_types": {
                    "lowercase": include_lowercase,
                    "uppercase": include_uppercase,
                    "numbers": include_numbers,
                    "symbols": include_symbols
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="password_generator",
            description="Generate secure passwords",
            category="security",
            parameters={
                "length": {"type": "integer", "description": "Password length", "default": 12},
                "include_uppercase": {"type": "boolean", "description": "Include uppercase", "default": True},
                "include_lowercase": {"type": "boolean", "description": "Include lowercase", "default": True},
                "include_numbers": {"type": "boolean", "description": "Include numbers", "default": True},
                "include_symbols": {"type": "boolean", "description": "Include symbols", "default": False}
            }
        ),
        password_generator_execute
    ))
    
    return tools


# ============================================================================
# MATHEMATICAL AND STATISTICAL TOOLS
# ============================================================================

def create_math_tools() -> List[ComprehensiveTool]:
    """Create mathematical and statistical computation tools"""
    tools = []
    
    # Statistical calculator
    def statistics_calculator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        data = args['data']
        operations = args.get('operations', ['mean', 'median', 'std'])
        
        try:
            if not isinstance(data, list) or not data:
                return {"success": False, "error": "Data must be a non-empty list"}
            
            # Convert to numbers
            numeric_data = []
            for item in data:
                try:
                    numeric_data.append(float(item))
                except (ValueError, TypeError):
                    continue
            
            if not numeric_data:
                return {"success": False, "error": "No numeric data found"}
            
            results = {}
            
            if 'mean' in operations:
                results['mean'] = statistics.mean(numeric_data)
            
            if 'median' in operations:
                results['median'] = statistics.median(numeric_data)
            
            if 'mode' in operations:
                try:
                    results['mode'] = statistics.mode(numeric_data)
                except statistics.StatisticsError:
                    results['mode'] = None  # No unique mode
            
            if 'std' in operations or 'stdev' in operations:
                if len(numeric_data) > 1:
                    results['standard_deviation'] = statistics.stdev(numeric_data)
                else:
                    results['standard_deviation'] = 0
            
            if 'variance' in operations:
                if len(numeric_data) > 1:
                    results['variance'] = statistics.variance(numeric_data)
                else:
                    results['variance'] = 0
            
            if 'range' in operations:
                results['range'] = max(numeric_data) - min(numeric_data)
                results['min'] = min(numeric_data)
                results['max'] = max(numeric_data)
            
            if 'quartiles' in operations:
                sorted_data = sorted(numeric_data)
                n = len(sorted_data)
                results['q1'] = sorted_data[n // 4] if n >= 4 else sorted_data[0]
                results['q2'] = statistics.median(sorted_data)
                results['q3'] = sorted_data[3 * n // 4] if n >= 4 else sorted_data[-1]
            
            return {
                "success": True,
                "data_count": len(numeric_data),
                "original_count": len(data),
                "statistics": results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="statistics_calculator",
            description="Calculate statistical measures for datasets",
            category="mathematics",
            parameters={
                "data": {"type": "array", "description": "Numeric data array"},
                "operations": {"type": "array", "description": "Statistical operations to perform"}
            }
        ),
        statistics_calculator_execute
    ))
    
    # Mathematical expression evaluator
    def math_evaluator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        expression = args['expression']
        variables = args.get('variables', {})
        
        try:
            # Safe mathematical evaluation
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow,
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'log10': math.log10, 'exp': math.exp,
                'pi': math.pi, 'e': math.e,
                'floor': math.floor, 'ceil': math.ceil,
                'degrees': math.degrees, 'radians': math.radians
            }
            
            # Add user variables
            allowed_names.update(variables)
            
            # Evaluate expression safely
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "result_type": type(result).__name__,
                "variables_used": list(variables.keys())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    tools.append(ComprehensiveTool(
        ToolMetadata(
            name="math_evaluator",
            description="Evaluate mathematical expressions safely",
            category="mathematics",
            parameters={
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
                "variables": {"type": "object", "description": "Variables for the expression"}
            }
        ),
        math_evaluator_execute
    ))
    
    return tools


# ============================================================================
# MAIN TOOLKIT ASSEMBLY
# ============================================================================

def get_comprehensive_toolkit() -> List[ComprehensiveTool]:
    """Get the complete comprehensive toolkit"""
    all_tools = []
    
    # Add all tool categories
    all_tools.extend(create_file_system_tools())
    all_tools.extend(create_web_tools())
    all_tools.extend(create_data_processing_tools())
    all_tools.extend(create_system_tools())
    all_tools.extend(create_creative_tools())
    all_tools.extend(create_communication_tools())
    all_tools.extend(create_security_tools())
    all_tools.extend(create_math_tools())
    
    return all_tools


def get_tools_by_category() -> Dict[str, List[ComprehensiveTool]]:
    """Get tools organized by category"""
    all_tools = get_comprehensive_toolkit()
    categories = {}
    
    for tool in all_tools:
        category = tool.category
        if category not in categories:
            categories[category] = []
        categories[category].append(tool)
    
    return categories


def get_tool_registry() -> Dict[str, Dict[str, Any]]:
    """Get a registry of all tools with their metadata"""
    all_tools = get_comprehensive_toolkit()
    registry = {}
    
    for tool in all_tools:
        registry[tool.name] = {
            "description": tool.description,
            "category": tool.category,
            "parameters": tool.metadata.parameters,
            "complexity": tool.metadata.complexity,
            "requires_network": tool.metadata.requires_network,
            "requires_filesystem": tool.metadata.requires_filesystem,
            "safety_level": tool.metadata.safety_level
        }
    
    return registry


if __name__ == "__main__":
    # Demo the comprehensive toolkit
    print("Comprehensive Agent Toolkit")
    print("=" * 50)
    
    tools_by_category = get_tools_by_category()
    
    for category, tools in tools_by_category.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    
    print(f"\nTotal tools available: {len(get_comprehensive_toolkit())}")
    print(f"Categories: {len(tools_by_category)}")