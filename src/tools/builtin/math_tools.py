"""
Mathematical computation tools for the autonomous agent.
"""

import math
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from ..interfaces import Tool
from ...core.models import ToolResult


class MathTool(Tool):
    """Base class for mathematical tools"""
    
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
        return "mathematics"
    
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


def create_math_tools() -> List[Tool]:
    """Create mathematical computation tools"""
    tools = []
    
    # Statistics calculator tool
    def statistics_calculator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        data = args['data']
        operations = args.get('operations', ['mean', 'median', 'std'])
        
        if not isinstance(data, list) or not data:
            raise ValueError("Data must be a non-empty list")
        
        # Convert to numbers
        numeric_data = []
        for item in data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if not numeric_data:
            raise ValueError("No numeric data found")
        
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
    
    tools.append(MathTool(
        "statistics_calculator",
        "Calculate statistical measures for datasets",
        statistics_calculator_execute
    ))
    
    # Mathematical expression evaluator tool
    def math_evaluator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        expression = args['expression']
        variables = args.get('variables', {})
        
        # Safe mathematical evaluation
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'pow': pow,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'exp': math.exp,
            'pi': math.pi, 'e': math.e,
            'floor': math.floor, 'ceil': math.ceil,
            'degrees': math.degrees, 'radians': math.radians,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh
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
    
    tools.append(MathTool(
        "math_evaluator",
        "Evaluate mathematical expressions safely",
        math_evaluator_execute
    ))
    
    # Number converter tool
    def number_converter_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        number = args['number']
        from_base = args.get('from_base', 10)
        to_base = args.get('to_base', 10)
        
        # Convert from source base to decimal
        if from_base == 10:
            decimal_value = int(number)
        elif from_base == 2:
            decimal_value = int(str(number), 2)
        elif from_base == 8:
            decimal_value = int(str(number), 8)
        elif from_base == 16:
            decimal_value = int(str(number), 16)
        else:
            decimal_value = int(str(number), from_base)
        
        # Convert from decimal to target base
        if to_base == 10:
            result = str(decimal_value)
        elif to_base == 2:
            result = bin(decimal_value)[2:]  # Remove '0b' prefix
        elif to_base == 8:
            result = oct(decimal_value)[2:]  # Remove '0o' prefix
        elif to_base == 16:
            result = hex(decimal_value)[2:]  # Remove '0x' prefix
        else:
            # Custom base conversion
            if decimal_value == 0:
                result = "0"
            else:
                digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                result = ""
                while decimal_value > 0:
                    result = digits[decimal_value % to_base] + result
                    decimal_value //= to_base
        
        return {
            "success": True,
            "original_number": str(number),
            "from_base": from_base,
            "to_base": to_base,
            "decimal_value": decimal_value,
            "converted_result": result
        }
    
    tools.append(MathTool(
        "number_converter",
        "Convert numbers between different bases (binary, octal, decimal, hexadecimal)",
        number_converter_execute
    ))
    
    # Geometry calculator tool
    def geometry_calculator_execute(args: Dict[str, Any]) -> Dict[str, Any]:
        shape = args['shape']
        dimensions = args['dimensions']
        
        if shape == 'circle':
            radius = dimensions['radius']
            area = math.pi * radius ** 2
            circumference = 2 * math.pi * radius
            return {
                "success": True,
                "shape": shape,
                "radius": radius,
                "area": area,
                "circumference": circumference
            }
        
        elif shape == 'rectangle':
            width = dimensions['width']
            height = dimensions['height']
            area = width * height
            perimeter = 2 * (width + height)
            diagonal = math.sqrt(width ** 2 + height ** 2)
            return {
                "success": True,
                "shape": shape,
                "width": width,
                "height": height,
                "area": area,
                "perimeter": perimeter,
                "diagonal": diagonal
            }
        
        elif shape == 'triangle':
            base = dimensions['base']
            height = dimensions['height']
            area = 0.5 * base * height
            
            # If all sides are provided, calculate perimeter
            if 'side1' in dimensions and 'side2' in dimensions and 'side3' in dimensions:
                perimeter = dimensions['side1'] + dimensions['side2'] + dimensions['side3']
            else:
                perimeter = None
            
            return {
                "success": True,
                "shape": shape,
                "base": base,
                "height": height,
                "area": area,
                "perimeter": perimeter
            }
        
        elif shape == 'sphere':
            radius = dimensions['radius']
            volume = (4/3) * math.pi * radius ** 3
            surface_area = 4 * math.pi * radius ** 2
            return {
                "success": True,
                "shape": shape,
                "radius": radius,
                "volume": volume,
                "surface_area": surface_area
            }
        
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    
    tools.append(MathTool(
        "geometry_calculator",
        "Calculate area, perimeter, volume for geometric shapes",
        geometry_calculator_execute
    ))
    
    return tools