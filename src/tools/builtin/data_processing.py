"""
Data processing tools for the autonomous agent.
"""

import json
import csv
import statistics
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from io import StringIO

from ..interfaces import Tool
from ...core.models import ToolResult


class DataProcessingTool(Tool):
    """Base class for data processing tools"""
    
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
        return "data_processing"
    
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


def create_data_tools() -> List[Tool]:
    """Create data processing tools"""
    tools = []
    
    # JSON processor tool
    def json_processor_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # parse, stringify, query, transform
        data = args['data']
        
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
                raise ValueError("Data must be a string for parsing")
        
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
                        raise KeyError(f"Path '{path}' not found")
            
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
                        func = transform['function']
                        if func == 'upper':
                            result = {k: str(v).upper() if isinstance(v, str) else v 
                                    for k, v in result.items()}
            
            return {
                "success": True,
                "result": result,
                "transformations_applied": len(transformations)
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(DataProcessingTool(
        "json_processor",
        "Process and manipulate JSON data",
        json_processor_execute
    ))
    
    # CSV processor tool
    def csv_processor_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # parse, stringify, analyze
        
        if operation == 'parse':
            csv_text = args['csv_text']
            delimiter = args.get('delimiter', ',')
            
            reader = csv.DictReader(StringIO(csv_text), delimiter=delimiter)
            rows = list(reader)
            
            return {
                "success": True,
                "data": rows,
                "row_count": len(rows),
                "columns": reader.fieldnames if rows else []
            }
        
        elif operation == 'stringify':
            data = args['data']
            delimiter = args.get('delimiter', ',')
            
            if not data:
                raise ValueError("No data provided")
            
            output = StringIO()
            fieldnames = data[0].keys() if isinstance(data[0], dict) else None
            
            if fieldnames:
                writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.writer(output, delimiter=delimiter)
                writer.writerows(data)
            
            return {
                "success": True,
                "csv_text": output.getvalue(),
                "rows_written": len(data)
            }
        
        elif operation == 'analyze':
            data = args['data']
            
            if not data:
                raise ValueError("No data provided")
            
            analysis = {
                "row_count": len(data),
                "column_count": len(data[0]) if data else 0,
                "columns": list(data[0].keys()) if isinstance(data[0], dict) else [],
                "sample_rows": data[:5]
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
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(DataProcessingTool(
        "csv_processor",
        "Process and analyze CSV data",
        csv_processor_execute
    ))
    
    # Text processor tool
    def text_processor_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        operation = args['operation']  # analyze, transform, extract
        text = args['text']
        
        if operation == 'analyze':
            words = text.split()
            sentences = text.split('.')
            paragraphs = text.split('\n\n')
            
            return {
                "success": True,
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "paragraph_count": len([p for p in paragraphs if p.strip()]),
                "character_count": len(text),
                "character_count_no_spaces": len(text.replace(' ', '')),
                "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "reading_time_minutes": len(words) / 200  # Average reading speed
            }
        
        elif operation == 'transform':
            transform_type = args['transform_type']
            
            if transform_type == 'uppercase':
                result = text.upper()
            elif transform_type == 'lowercase':
                result = text.lower()
            elif transform_type == 'title_case':
                result = text.title()
            elif transform_type == 'remove_extra_spaces':
                result = re.sub(r'\s+', ' ', text).strip()
            elif transform_type == 'remove_punctuation':
                result = re.sub(r'[^\w\s]', '', text)
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
            
            return {
                "success": True,
                "original_text": text,
                "transformed_text": result,
                "transform_type": transform_type
            }
        
        elif operation == 'extract':
            extract_type = args['extract_type']
            
            if extract_type == 'emails':
                pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                matches = re.findall(pattern, text)
            elif extract_type == 'urls':
                pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                matches = re.findall(pattern, text)
            elif extract_type == 'phone_numbers':
                pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
                matches = re.findall(pattern, text)
            elif extract_type == 'numbers':
                pattern = r'\b\d+\.?\d*\b'
                matches = re.findall(pattern, text)
            else:
                raise ValueError(f"Unknown extract type: {extract_type}")
            
            return {
                "success": True,
                "extract_type": extract_type,
                "matches": matches,
                "match_count": len(matches)
            }
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    tools.append(DataProcessingTool(
        "text_processor",
        "Analyze, transform, and extract data from text",
        text_processor_execute
    ))
    
    # Data validator tool
    def data_validator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        data = args['data']
        validation_rules = args['validation_rules']
        
        results = []
        
        for rule in validation_rules:
            rule_type = rule['type']
            field = rule.get('field')
            
            if rule_type == 'required':
                # Check if field exists and is not empty
                if isinstance(data, dict):
                    is_valid = field in data and data[field] is not None and str(data[field]).strip() != ''
                else:
                    is_valid = False
                
                results.append({
                    "rule": rule,
                    "valid": is_valid,
                    "message": f"Field '{field}' is required" if not is_valid else "Valid"
                })
            
            elif rule_type == 'type':
                # Check data type
                expected_type = rule['expected_type']
                if isinstance(data, dict) and field in data:
                    value = data[field]
                    if expected_type == 'string':
                        is_valid = isinstance(value, str)
                    elif expected_type == 'number':
                        is_valid = isinstance(value, (int, float))
                    elif expected_type == 'boolean':
                        is_valid = isinstance(value, bool)
                    else:
                        is_valid = False
                else:
                    is_valid = False
                
                results.append({
                    "rule": rule,
                    "valid": is_valid,
                    "message": f"Field '{field}' must be {expected_type}" if not is_valid else "Valid"
                })
            
            elif rule_type == 'range':
                # Check numeric range
                if isinstance(data, dict) and field in data:
                    value = data[field]
                    min_val = rule.get('min')
                    max_val = rule.get('max')
                    
                    try:
                        num_value = float(value)
                        is_valid = True
                        if min_val is not None and num_value < min_val:
                            is_valid = False
                        if max_val is not None and num_value > max_val:
                            is_valid = False
                    except (ValueError, TypeError):
                        is_valid = False
                else:
                    is_valid = False
                
                results.append({
                    "rule": rule,
                    "valid": is_valid,
                    "message": f"Field '{field}' must be between {min_val} and {max_val}" if not is_valid else "Valid"
                })
        
        all_valid = all(result['valid'] for result in results)
        
        return {
            "success": True,
            "all_valid": all_valid,
            "validation_results": results,
            "total_rules": len(validation_rules),
            "passed_rules": sum(1 for result in results if result['valid'])
        }
    
    tools.append(DataProcessingTool(
        "data_validator",
        "Validate data against specified rules",
        data_validator_execute
    ))
    
    return tools