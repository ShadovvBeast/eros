"""
Web operation tools for the autonomous agent.
"""

import urllib.request
import urllib.parse
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..interfaces import Tool
from ...core.models import ToolResult


class WebOperationTool(Tool):
    """Base class for web operation tools"""
    
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
        return "web_operations"
    
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


def create_web_tools() -> List[Tool]:
    """Create web operation tools"""
    tools = []
    
    # HTTP request tool
    def http_request_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        url = args['url']
        method = args.get('method', 'GET').upper()
        headers = args.get('headers', {})
        data = args.get('data', None)
        timeout = args.get('timeout', 30)
        
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
    
    def http_request_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if 'url' not in args:
            return False, "Missing required argument: url"
        if not isinstance(args['url'], str):
            return False, "URL must be a string"
        return True, None
    
    tools.append(WebOperationTool(
        "http_request",
        "Make HTTP requests to web services",
        http_request_execute,
        http_request_validate
    ))
    
    # Web scraper tool
    def web_scrape_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        url = args['url']
        extract_links = args.get('extract_links', False)
        extract_images = args.get('extract_images', False)
        extract_text = args.get('extract_text', True)
        
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
    
    tools.append(WebOperationTool(
        "web_scrape",
        "Scrape content from web pages",
        web_scrape_execute
    ))
    
    # URL utilities tool
    def url_utilities_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # parse, build, encode, decode
        
        if operation == 'parse':
            url = args['url']
            parsed = urllib.parse.urlparse(url)
            return {
                "success": True,
                "scheme": parsed.scheme,
                "netloc": parsed.netloc,
                "path": parsed.path,
                "params": parsed.params,
                "query": parsed.query,
                "fragment": parsed.fragment,
                "hostname": parsed.hostname,
                "port": parsed.port
            }
        
        elif operation == 'build':
            components = args['components']
            url = urllib.parse.urlunparse((
                components.get('scheme', 'https'),
                components.get('netloc', ''),
                components.get('path', ''),
                components.get('params', ''),
                components.get('query', ''),
                components.get('fragment', '')
            ))
            return {
                "success": True,
                "url": url
            }
        
        elif operation == 'encode':
            text = args['text']
            encoded = urllib.parse.quote(text)
            return {
                "success": True,
                "original": text,
                "encoded": encoded
            }
        
        elif operation == 'decode':
            text = args['text']
            decoded = urllib.parse.unquote(text)
            return {
                "success": True,
                "original": text,
                "decoded": decoded
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(WebOperationTool(
        "url_utilities",
        "Parse, build, encode, and decode URLs",
        url_utilities_execute
    ))
    
    # Download file tool
    def download_file_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        url = args['url']
        output_path = args.get('output_path', None)
        
        # If no output path specified, use filename from URL
        if not output_path:
            parsed_url = urllib.parse.urlparse(url)
            filename = parsed_url.path.split('/')[-1] or 'downloaded_file'
            output_path = filename
        
        # Download the file
        urllib.request.urlretrieve(url, output_path)
        
        # Get file info
        import os
        file_size = os.path.getsize(output_path)
        
        return {
            "success": True,
            "url": url,
            "output_path": output_path,
            "file_size": file_size,
            "downloaded": True
        }
    
    tools.append(WebOperationTool(
        "download_file",
        "Download files from URLs",
        download_file_execute
    ))
    
    return tools