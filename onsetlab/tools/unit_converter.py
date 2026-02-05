"""UnitConverter tool for converting between units."""

from typing import Any, Dict, Optional
from .base import BaseTool


class UnitConverter(BaseTool):
    """
    Convert between common units: length, weight, temperature, volume, speed, data.
    """
    
    name = "UnitConverter"
    description = "Convert between units: length (km/mi/m/ft), weight (kg/lb/oz), temperature (C/F/K), volume (L/gal), speed (km/h/mph), data (GB/MB/KB)."
    
    # Conversion factors to base units
    CONVERSIONS = {
        # Length - base: meters
        "length": {
            "m": 1.0,
            "meter": 1.0,
            "meters": 1.0,
            "km": 1000.0,
            "kilometer": 1000.0,
            "kilometers": 1000.0,
            "cm": 0.01,
            "centimeter": 0.01,
            "centimeters": 0.01,
            "mm": 0.001,
            "millimeter": 0.001,
            "millimeters": 0.001,
            "mi": 1609.344,
            "mile": 1609.344,
            "miles": 1609.344,
            "ft": 0.3048,
            "foot": 0.3048,
            "feet": 0.3048,
            "in": 0.0254,
            "inch": 0.0254,
            "inches": 0.0254,
            "yd": 0.9144,
            "yard": 0.9144,
            "yards": 0.9144,
        },
        # Weight - base: kilograms
        "weight": {
            "kg": 1.0,
            "kilogram": 1.0,
            "kilograms": 1.0,
            "g": 0.001,
            "gram": 0.001,
            "grams": 0.001,
            "mg": 0.000001,
            "milligram": 0.000001,
            "milligrams": 0.000001,
            "lb": 0.453592,
            "lbs": 0.453592,
            "pound": 0.453592,
            "pounds": 0.453592,
            "oz": 0.0283495,
            "ounce": 0.0283495,
            "ounces": 0.0283495,
            "st": 6.35029,
            "stone": 6.35029,
        },
        # Volume - base: liters
        "volume": {
            "l": 1.0,
            "liter": 1.0,
            "liters": 1.0,
            "litre": 1.0,
            "litres": 1.0,
            "ml": 0.001,
            "milliliter": 0.001,
            "milliliters": 0.001,
            "gal": 3.78541,
            "gallon": 3.78541,
            "gallons": 3.78541,
            "qt": 0.946353,
            "quart": 0.946353,
            "quarts": 0.946353,
            "pt": 0.473176,
            "pint": 0.473176,
            "pints": 0.473176,
            "cup": 0.236588,
            "cups": 0.236588,
            "floz": 0.0295735,
            "fl oz": 0.0295735,
            "fluid ounce": 0.0295735,
        },
        # Speed - base: m/s
        "speed": {
            "m/s": 1.0,
            "mps": 1.0,
            "km/h": 0.277778,
            "kmh": 0.277778,
            "kph": 0.277778,
            "mph": 0.44704,
            "mi/h": 0.44704,
            "ft/s": 0.3048,
            "fps": 0.3048,
            "knot": 0.514444,
            "knots": 0.514444,
        },
        # Data - base: bytes
        "data": {
            "b": 1.0,
            "byte": 1.0,
            "bytes": 1.0,
            "kb": 1024.0,
            "kilobyte": 1024.0,
            "kilobytes": 1024.0,
            "mb": 1048576.0,
            "megabyte": 1048576.0,
            "megabytes": 1048576.0,
            "gb": 1073741824.0,
            "gigabyte": 1073741824.0,
            "gigabytes": 1073741824.0,
            "tb": 1099511627776.0,
            "terabyte": 1099511627776.0,
            "terabytes": 1099511627776.0,
        },
    }
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The numeric value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit (e.g., 'km', 'miles', 'kg', 'lb', 'celsius', 'fahrenheit')"
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit to convert to"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    
    def execute(self, value: float, from_unit: str, to_unit: str) -> str:
        """Convert value from one unit to another."""
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        
        # Handle temperature separately (not linear conversion)
        if self._is_temperature(from_unit) or self._is_temperature(to_unit):
            return self._convert_temperature(value, from_unit, to_unit)
        
        # Find the category for both units
        from_category = self._find_category(from_unit)
        to_category = self._find_category(to_unit)
        
        if not from_category:
            return f"Error: Unknown unit '{from_unit}'"
        if not to_category:
            return f"Error: Unknown unit '{to_unit}'"
        if from_category != to_category:
            return f"Error: Cannot convert {from_unit} ({from_category}) to {to_unit} ({to_category})"
        
        # Convert: value -> base unit -> target unit
        conversions = self.CONVERSIONS[from_category]
        base_value = value * conversions[from_unit]
        result = base_value / conversions[to_unit]
        
        # Format result nicely
        if result == int(result):
            result_str = str(int(result))
        elif abs(result) >= 0.01:
            result_str = f"{result:.4f}".rstrip('0').rstrip('.')
        else:
            result_str = f"{result:.6g}"
        
        return f"{value} {from_unit} = {result_str} {to_unit}"
    
    def _find_category(self, unit: str) -> Optional[str]:
        """Find which category a unit belongs to."""
        for category, units in self.CONVERSIONS.items():
            if unit in units:
                return category
        return None
    
    def _is_temperature(self, unit: str) -> bool:
        """Check if unit is a temperature unit."""
        temp_units = ['c', 'celsius', 'f', 'fahrenheit', 'k', 'kelvin']
        return unit in temp_units
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> str:
        """Convert temperature (special case - not linear)."""
        # Normalize unit names
        temp_map = {
            'c': 'celsius', 'celsius': 'celsius',
            'f': 'fahrenheit', 'fahrenheit': 'fahrenheit',
            'k': 'kelvin', 'kelvin': 'kelvin'
        }
        
        from_temp = temp_map.get(from_unit)
        to_temp = temp_map.get(to_unit)
        
        if not from_temp:
            return f"Error: Unknown temperature unit '{from_unit}'"
        if not to_temp:
            return f"Error: Unknown temperature unit '{to_unit}'"
        
        # Convert to Celsius first
        if from_temp == 'celsius':
            celsius = value
        elif from_temp == 'fahrenheit':
            celsius = (value - 32) * 5 / 9
        elif from_temp == 'kelvin':
            celsius = value - 273.15
        
        # Convert from Celsius to target
        if to_temp == 'celsius':
            result = celsius
        elif to_temp == 'fahrenheit':
            result = celsius * 9 / 5 + 32
        elif to_temp == 'kelvin':
            result = celsius + 273.15
        
        # Format result
        if result == int(result):
            result_str = str(int(result))
        else:
            result_str = f"{result:.2f}".rstrip('0').rstrip('.')
        
        # Use symbols for display
        symbols = {'celsius': '°C', 'fahrenheit': '°F', 'kelvin': 'K'}
        from_sym = symbols.get(from_temp, from_unit)
        to_sym = symbols.get(to_temp, to_unit)
        
        return f"{value}{from_sym} = {result_str}{to_sym}"
