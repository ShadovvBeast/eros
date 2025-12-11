"""
Security tools for the autonomous agent.
"""

import hashlib
import base64
import random
import string
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..interfaces import Tool
from ...models import ToolResult


class SecurityTool(Tool):
    """Base class for security tools"""
    
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
        return "security"
    
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


def create_security_tools() -> List[Tool]:
    """Create security tools"""
    tools = []
    
    # Hash generator tool
    def hash_generator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        data = args['data']
        algorithm = args.get('algorithm', 'sha256')
        
        # Convert data to bytes
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
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        hex_hash = hash_obj.hexdigest()
        
        return {
            "success": True,
            "original_data": str(data)[:100] + "..." if len(str(data)) > 100 else str(data),
            "hash": hex_hash,
            "algorithm": algorithm,
            "hash_length": len(hex_hash)
        }
    
    tools.append(SecurityTool(
        "hash_generator",
        "Generate cryptographic hashes (MD5, SHA1, SHA256, SHA512)",
        hash_generator_execute
    ))
    
    # Base64 processor tool
    def base64_processor_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # encode, decode
        data = args['data']
        
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
                raise ValueError(f"Invalid base64 data: {str(decode_error)}")
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(SecurityTool(
        "base64_processor",
        "Encode and decode base64 data",
        base64_processor_execute
    ))
    
    # Password generator tool
    def password_generator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        length = args.get('length', 12)
        include_uppercase = args.get('include_uppercase', True)
        include_lowercase = args.get('include_lowercase', True)
        include_numbers = args.get('include_numbers', True)
        include_symbols = args.get('include_symbols', False)
        
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
            raise ValueError("No character types selected")
        
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
    
    tools.append(SecurityTool(
        "password_generator",
        "Generate secure passwords with customizable criteria",
        password_generator_execute
    ))
    
    return tools