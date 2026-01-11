"""
Meta-tools for tool discovery and self-development.

These tools give the agent the ability to:
1. Search for and discover new tools from external sources (APIs, registries, etc.)
2. Develop new tools and integrate them into itself
"""

import json
import time
import hashlib
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime

from ..interfaces import Tool
from ...core.models import ToolResult


class MetaTool(Tool):
    """Base class for meta-tools"""
    
    def __init__(self, name: str, description: str, execute_func: Callable, 
                 validate_func: Optional[Callable] = None, tool_layer=None):
        self._name = name
        self._description = description
        self._execute_func = execute_func
        self._validate_func = validate_func or self._default_validate
        self._tool_layer = tool_layer  # Reference to tool layer for self-modification
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def category(self) -> str:
        return "meta_tools"
    
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


# Global registry for dynamically created tools
_dynamic_tools_registry: Dict[str, Dict[str, Any]] = {}


def create_meta_tools(tool_layer=None) -> List[Tool]:
    """Create meta-tools for tool discovery and development."""
    tools = []
    
    # =========================================================================
    # TOOL 1: Tool Discovery - Search for tools from external sources
    # =========================================================================
    def tool_finder_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for tools from multiple sources:
        - NPM registry (for Node.js tools/packages)
        - PyPI (for Python packages)
        - GitHub (for repositories)
        - Public API directories
        """
        query = args.get('query', '')
        source = args.get('source', 'all')  # 'npm', 'pypi', 'github', 'apis', 'all'
        limit = args.get('limit', 10)
        
        results = {
            'query': query,
            'timestamp': int(time.time()),
            'sources_searched': [],
            'tools_found': [],
            'total_count': 0
        }
        
        # Search NPM Registry
        if source in ['npm', 'all']:
            try:
                npm_url = f"https://registry.npmjs.org/-/v1/search?text={urllib.parse.quote(query)}&size={limit}"
                with urllib.request.urlopen(npm_url, timeout=10) as response:
                    npm_data = json.loads(response.read().decode('utf-8'))
                    
                results['sources_searched'].append('npm')
                for pkg in npm_data.get('objects', [])[:limit]:
                    package = pkg.get('package', {})
                    results['tools_found'].append({
                        'name': package.get('name', ''),
                        'description': package.get('description', '')[:200] if package.get('description') else '',
                        'version': package.get('version', ''),
                        'source': 'npm',
                        'install_command': f"npm install {package.get('name', '')}",
                        'url': f"https://www.npmjs.com/package/{package.get('name', '')}",
                        'keywords': package.get('keywords', [])[:5]
                    })
            except Exception as e:
                results['npm_error'] = str(e)
        
        # Search PyPI
        if source in ['pypi', 'all']:
            try:
                pypi_url = f"https://pypi.org/pypi/{urllib.parse.quote(query)}/json"
                try:
                    with urllib.request.urlopen(pypi_url, timeout=10) as response:
                        pypi_data = json.loads(response.read().decode('utf-8'))
                        info = pypi_data.get('info', {})
                        
                        results['sources_searched'].append('pypi')
                        results['tools_found'].append({
                            'name': info.get('name', ''),
                            'description': info.get('summary', '')[:200] if info.get('summary') else '',
                            'version': info.get('version', ''),
                            'source': 'pypi',
                            'install_command': f"pip install {info.get('name', '')}",
                            'url': info.get('project_url', '') or f"https://pypi.org/project/{info.get('name', '')}/",
                            'keywords': info.get('keywords', '').split(',')[:5] if info.get('keywords') else []
                        })
                except urllib.error.HTTPError:
                    # Try search API instead
                    search_url = f"https://pypi.org/search/?q={urllib.parse.quote(query)}"
                    results['pypi_search_url'] = search_url
                    results['sources_searched'].append('pypi_search')
            except Exception as e:
                results['pypi_error'] = str(e)
        
        # Search GitHub
        if source in ['github', 'all']:
            try:
                github_url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}+topic:tool&sort=stars&per_page={limit}"
                req = urllib.request.Request(github_url, headers={'Accept': 'application/vnd.github.v3+json'})
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    github_data = json.loads(response.read().decode('utf-8'))
                    
                results['sources_searched'].append('github')
                for repo in github_data.get('items', [])[:limit]:
                    results['tools_found'].append({
                        'name': repo.get('name', ''),
                        'description': repo.get('description', '')[:200] if repo.get('description') else '',
                        'version': 'latest',
                        'source': 'github',
                        'install_command': f"git clone {repo.get('clone_url', '')}",
                        'url': repo.get('html_url', ''),
                        'stars': repo.get('stargazers_count', 0),
                        'language': repo.get('language', ''),
                        'topics': repo.get('topics', [])[:5]
                    })
            except Exception as e:
                results['github_error'] = str(e)
        
        # Search Public APIs Directory
        if source in ['apis', 'all']:
            try:
                # Use the public-apis project
                apis_url = "https://api.publicapis.org/entries"
                params = f"?title={urllib.parse.quote(query)}" if query else ""
                
                with urllib.request.urlopen(apis_url + params, timeout=10) as response:
                    apis_data = json.loads(response.read().decode('utf-8'))
                    
                results['sources_searched'].append('public_apis')
                for api in (apis_data.get('entries', []) or [])[:limit]:
                    results['tools_found'].append({
                        'name': api.get('API', ''),
                        'description': api.get('Description', '')[:200],
                        'version': 'api',
                        'source': 'public_api',
                        'url': api.get('Link', ''),
                        'category': api.get('Category', ''),
                        'auth': api.get('Auth', 'None'),
                        'https': api.get('HTTPS', False),
                        'cors': api.get('Cors', 'unknown')
                    })
            except Exception as e:
                results['apis_error'] = str(e)
        
        results['total_count'] = len(results['tools_found'])
        
        # Add recommendations based on results
        if results['tools_found']:
            results['recommendations'] = [
                f"Found {results['total_count']} potential tools/APIs for '{query}'",
                "Use 'tool_developer' to create a wrapper for any of these tools",
                "Consider the tool's documentation and API stability before integration"
            ]
        else:
            results['recommendations'] = [
                f"No tools found for '{query}'",
                "Try broader search terms or different sources",
                "Consider developing a custom tool using 'tool_developer'"
            ]
        
        return results
    
    def tool_finder_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if 'query' not in args:
            return False, "Missing required argument: query"
        if not isinstance(args['query'], str):
            return False, "Argument 'query' must be a string"
        return True, None
    
    tools.append(MetaTool(
        name="tool_finder",
        description="Search for tools and APIs from NPM, PyPI, GitHub, and public API directories",
        execute_func=tool_finder_execute,
        validate_func=tool_finder_validate,
        tool_layer=tool_layer
    ))
    
    # =========================================================================
    # TOOL 2: Tool Developer - Create and integrate new tools
    # =========================================================================
    def tool_developer_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Develop a new tool based on specification and optionally integrate it.
        
        Supports creating tools that:
        - Wrap external APIs
        - Process data in specific ways
        - Combine existing tool capabilities
        - Execute custom logic
        """
        global _dynamic_tools_registry
        
        spec = args.get('specification', {})
        action = args.get('action', 'create')  # 'create', 'integrate', 'list', 'remove'
        
        if action == 'list':
            # List all dynamically created tools
            return {
                'action': 'list',
                'dynamic_tools': list(_dynamic_tools_registry.keys()),
                'count': len(_dynamic_tools_registry),
                'details': {
                    name: {
                        'description': info.get('description', ''),
                        'category': info.get('category', ''),
                        'created_at': info.get('created_at', ''),
                        'integrated': info.get('integrated', False)
                    }
                    for name, info in _dynamic_tools_registry.items()
                }
            }
        
        if action == 'remove':
            tool_name = args.get('tool_name', '')
            if tool_name in _dynamic_tools_registry:
                del _dynamic_tools_registry[tool_name]
                return {
                    'action': 'remove',
                    'tool_name': tool_name,
                    'success': True,
                    'message': f"Tool '{tool_name}' removed from registry"
                }
            return {
                'action': 'remove',
                'tool_name': tool_name,
                'success': False,
                'message': f"Tool '{tool_name}' not found in registry"
            }
        
        # Validate specification
        if not spec:
            return {
                'success': False,
                'error': 'No specification provided',
                'required_fields': ['name', 'description', 'category', 'logic_type'],
                'example_spec': {
                    'name': 'my_custom_tool',
                    'description': 'A tool that does something useful',
                    'category': 'custom',
                    'logic_type': 'api_wrapper',  # or 'data_processor', 'combiner', 'custom'
                    'config': {
                        'api_url': 'https://api.example.com',
                        'method': 'GET',
                        'headers': {}
                    }
                }
            }
        
        tool_name = spec.get('name', f"dynamic_tool_{int(time.time())}")
        description = spec.get('description', 'A dynamically created tool')
        category = spec.get('category', 'dynamic')
        logic_type = spec.get('logic_type', 'custom')
        config = spec.get('config', {})
        
        # Generate tool ID
        tool_id = hashlib.md5(f"{tool_name}_{time.time()}".encode()).hexdigest()[:8]
        
        # Create the tool based on logic type
        tool_code = None
        execute_func = None
        
        if logic_type == 'api_wrapper':
            # Create an API wrapper tool
            api_url = config.get('api_url', '')
            method = config.get('method', 'GET')
            headers = config.get('headers', {})
            
            def create_api_wrapper(url, http_method, http_headers):
                def api_execute(inner_args: Dict[str, Any]) -> Dict[str, Any]:
                    try:
                        # Build URL with query params for GET
                        request_url = url
                        data = None
                        
                        if http_method == 'GET' and inner_args:
                            params = urllib.parse.urlencode(inner_args)
                            request_url = f"{url}?{params}"
                        elif http_method in ['POST', 'PUT', 'PATCH']:
                            data = json.dumps(inner_args).encode('utf-8')
                            http_headers['Content-Type'] = 'application/json'
                        
                        req = urllib.request.Request(
                            request_url,
                            data=data,
                            headers=http_headers,
                            method=http_method
                        )
                        
                        with urllib.request.urlopen(req, timeout=30) as response:
                            content = response.read().decode('utf-8')
                            try:
                                result = json.loads(content)
                            except json.JSONDecodeError:
                                result = {'raw_response': content}
                            
                            return {
                                'success': True,
                                'status_code': response.getcode(),
                                'data': result
                            }
                    except Exception as e:
                        return {
                            'success': False,
                            'error': str(e)
                        }
                return api_execute
            
            execute_func = create_api_wrapper(api_url, method, headers)
            tool_code = f"""
# API Wrapper Tool: {tool_name}
# URL: {api_url}
# Method: {method}
def execute(args):
    # Makes {method} request to {api_url}
    # Returns JSON response
    pass
"""
        
        elif logic_type == 'data_processor':
            # Create a data processing tool
            processor_type = config.get('processor_type', 'transform')
            
            def create_data_processor(proc_type):
                def data_execute(inner_args: Dict[str, Any]) -> Dict[str, Any]:
                    data = inner_args.get('data', inner_args)
                    
                    if proc_type == 'transform':
                        # Apply transformations
                        transforms = inner_args.get('transforms', [])
                        result = data
                        for t in transforms:
                            if t == 'uppercase' and isinstance(result, str):
                                result = result.upper()
                            elif t == 'lowercase' and isinstance(result, str):
                                result = result.lower()
                            elif t == 'reverse' and isinstance(result, (str, list)):
                                result = result[::-1]
                            elif t == 'sort' and isinstance(result, list):
                                result = sorted(result)
                        return {'transformed': result, 'transforms_applied': transforms}
                    
                    elif proc_type == 'filter':
                        # Filter data
                        condition = inner_args.get('condition', {})
                        if isinstance(data, list):
                            filtered = [item for item in data if all(
                                item.get(k) == v for k, v in condition.items()
                            ) if isinstance(item, dict)]
                            return {'filtered': filtered, 'count': len(filtered)}
                        return {'filtered': data, 'count': 1}
                    
                    elif proc_type == 'aggregate':
                        # Aggregate data
                        if isinstance(data, list):
                            numeric = [x for x in data if isinstance(x, (int, float))]
                            return {
                                'count': len(data),
                                'sum': sum(numeric) if numeric else 0,
                                'avg': sum(numeric) / len(numeric) if numeric else 0,
                                'min': min(numeric) if numeric else None,
                                'max': max(numeric) if numeric else None
                            }
                        return {'data': data, 'type': type(data).__name__}
                    
                    return {'processed': data, 'processor_type': proc_type}
                return data_execute
            
            execute_func = create_data_processor(processor_type)
            tool_code = f"""
# Data Processor Tool: {tool_name}
# Type: {processor_type}
def execute(args):
    # Processes data using {processor_type} logic
    pass
"""
        
        elif logic_type == 'combiner':
            # Create a tool that combines other tools
            tool_sequence = config.get('tool_sequence', [])
            
            def create_combiner(sequence):
                def combiner_execute(inner_args: Dict[str, Any]) -> Dict[str, Any]:
                    results = []
                    current_input = inner_args
                    
                    for step in sequence:
                        step_name = step.get('tool', 'unknown')
                        step_args = step.get('args', {})
                        # Merge current input with step args
                        merged_args = {**current_input, **step_args}
                        
                        results.append({
                            'step': step_name,
                            'input': merged_args,
                            'note': f"Would execute {step_name} here"
                        })
                        
                        # In a real implementation, this would call the actual tool
                        current_input = merged_args
                    
                    return {
                        'pipeline_results': results,
                        'steps_executed': len(sequence),
                        'final_output': current_input
                    }
                return combiner_execute
            
            execute_func = create_combiner(tool_sequence)
            tool_code = f"""
# Combiner Tool: {tool_name}
# Sequence: {[s.get('tool') for s in tool_sequence]}
def execute(args):
    # Executes tools in sequence, passing output to next input
    pass
"""
        
        else:  # custom
            # Create a custom tool with basic functionality
            custom_logic = config.get('logic', 'echo')
            
            def create_custom(logic):
                def custom_execute(inner_args: Dict[str, Any]) -> Dict[str, Any]:
                    if logic == 'echo':
                        return {'echo': inner_args, 'timestamp': int(time.time())}
                    elif logic == 'timestamp':
                        return {'timestamp': int(time.time()), 'iso': datetime.now().isoformat()}
                    elif logic == 'hash':
                        data = str(inner_args)
                        return {
                            'md5': hashlib.md5(data.encode()).hexdigest(),
                            'sha256': hashlib.sha256(data.encode()).hexdigest()
                        }
                    elif logic == 'format':
                        template = inner_args.get('template', '{data}')
                        data = inner_args.get('data', {})
                        try:
                            formatted = template.format(**data) if isinstance(data, dict) else template.format(data=data)
                            return {'formatted': formatted}
                        except Exception as e:
                            return {'error': str(e), 'template': template}
                    else:
                        return {'input': inner_args, 'logic': logic, 'processed': True}
                return custom_execute
            
            execute_func = create_custom(custom_logic)
            tool_code = f"""
# Custom Tool: {tool_name}
# Logic: {custom_logic}
def execute(args):
    # Custom logic: {custom_logic}
    pass
"""
        
        # Store in registry
        _dynamic_tools_registry[tool_name] = {
            'id': tool_id,
            'name': tool_name,
            'description': description,
            'category': category,
            'logic_type': logic_type,
            'config': config,
            'code': tool_code,
            'execute_func': execute_func,
            'created_at': datetime.now().isoformat(),
            'integrated': False
        }
        
        result = {
            'action': action,
            'success': True,
            'tool_id': tool_id,
            'tool_name': tool_name,
            'description': description,
            'category': category,
            'logic_type': logic_type,
            'created_at': datetime.now().isoformat(),
            'code_preview': tool_code,
            'status': 'created'
        }
        
        # If action is 'integrate', also register the tool
        if action == 'integrate' and tool_layer is not None:
            try:
                from ..tool_layer import MCPTool
                
                new_tool = MCPTool(
                    name=tool_name,
                    description=description,
                    category=category if category in tool_layer.config.allowed_tool_types else 'communication',
                    execute_func=execute_func
                )
                
                tool_layer.register_tool(new_tool)
                _dynamic_tools_registry[tool_name]['integrated'] = True
                
                result['status'] = 'integrated'
                result['message'] = f"Tool '{tool_name}' created and integrated into the agent"
                result['available_in'] = tool_layer.get_available_tools()
            except Exception as e:
                result['integration_error'] = str(e)
                result['status'] = 'created_not_integrated'
                result['message'] = f"Tool created but integration failed: {e}"
        else:
            result['message'] = f"Tool '{tool_name}' created. Use action='integrate' to add it to the agent."
        
        return result
    
    def tool_developer_validate(args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        action = args.get('action', 'create')
        if action in ['create', 'integrate']:
            if 'specification' not in args:
                return False, "Missing required argument: specification"
        elif action == 'remove':
            if 'tool_name' not in args:
                return False, "Missing required argument: tool_name for remove action"
        return True, None
    
    tools.append(MetaTool(
        name="tool_developer",
        description="Create, develop, and integrate new tools into the agent. Supports API wrappers, data processors, tool combiners, and custom logic.",
        execute_func=tool_developer_execute,
        validate_func=tool_developer_validate,
        tool_layer=tool_layer
    ))
    
    return tools


def get_dynamic_tools_registry() -> Dict[str, Dict[str, Any]]:
    """Get the current dynamic tools registry."""
    return _dynamic_tools_registry.copy()
