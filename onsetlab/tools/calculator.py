"""Calculator tool for math operations."""

import math
import re
from typing import Any, Dict

from .base import BaseTool


class Calculator(BaseTool):
    """
    Calculator tool for evaluating mathematical expressions.
    
    Supports basic arithmetic, percentages, and common math functions.
    """
    
    name = "Calculator"
    description = "Evaluate mathematical expressions. Supports +, -, *, /, %, ^, sqrt, sin, cos, tan, log, abs, round, floor, ceil."
    
    # Safe math functions available in expressions
    SAFE_FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'pow': pow,
        'pi': math.pi,
        'e': math.e,
    }
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '15 * 0.20', 'sqrt(16)', '100 + 15%')"
                }
            },
            "required": ["expression"]
        }
    
    def execute(self, expression: str) -> str:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Math expression as string.
            
        Returns:
            Result as string.
        """
        try:
            # Clean up the expression
            expr = expression.strip()
            
            # Handle percentage notation (e.g., "100 + 15%" -> "100 + 100*0.15")
            expr = self._handle_percentages(expr)
            
            # Replace ^ with ** for exponentiation
            expr = expr.replace('^', '**')
            
            # Validate expression (only allow safe characters)
            if not self._is_safe_expression(expr):
                return f"Error: Invalid characters in expression"
            
            # Evaluate with safe functions
            result = eval(expr, {"__builtins__": {}}, self.SAFE_FUNCTIONS)
            
            # Format result
            if isinstance(result, float):
                # Round to reasonable precision
                if result == int(result):
                    return str(int(result))
                return str(round(result, 10))
            return str(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _handle_percentages(self, expr: str) -> str:
        """Convert percentage notation to decimal multiplication."""
        # Pattern: number followed by % (e.g., "15%" -> "0.15")
        # Handle "X + Y%" as "X + X * Y/100" and "X * Y%" as "X * Y/100"
        
        # Simple case: standalone percentage (e.g., "15%")
        expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)
        
        return expr
    
    def _is_safe_expression(self, expr: str) -> bool:
        """Check if expression only contains safe characters."""
        # Allow: digits, operators, parentheses, spaces, dots, function names
        allowed_pattern = r'^[\d\s\+\-\*\/\.\(\)\,a-z_]+$'
        return bool(re.match(allowed_pattern, expr, re.IGNORECASE))
