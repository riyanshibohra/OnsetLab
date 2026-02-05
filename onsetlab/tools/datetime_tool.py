"""DateTime tool for date and time operations."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import calendar
import re

from .base import BaseTool


class DateTime(BaseTool):
    """
    DateTime tool for getting current time and date calculations.
    """
    
    name = "DateTime"
    description = "Get current date/time, day of week for a date, add days to a date, or calculate days between dates."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": ["now", "day_of_week", "add_days", "difference"]
                },
                "date": {
                    "type": "string",
                    "description": "Date (accepts: 2000-01-01, January 1 2000, Jan 1 2000, etc.)"
                },
                "date2": {
                    "type": "string",
                    "description": "Second date for difference calculation"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to add (negative to subtract)"
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC'). Default: local."
                }
            },
            "required": ["operation"]
        }
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format support."""
        if not date_str:
            return None
        
        # Clean up the string
        date_str = date_str.strip()
        date_lower = date_str.lower()
        
        # Handle special keywords
        if date_lower in ["now", "today", "current"]:
            return datetime.now()
        if date_lower == "yesterday":
            return datetime.now() - timedelta(days=1)
        if date_lower == "tomorrow":
            return datetime.now() + timedelta(days=1)
        
        # Common date formats to try
        formats = [
            "%Y-%m-%d",           # 2000-01-01
            "%B %d, %Y",          # January 1, 2000
            "%B %d %Y",           # January 1 2000
            "%b %d, %Y",          # Jan 1, 2000
            "%b %d %Y",           # Jan 1 2000
            "%d %B %Y",           # 1 January 2000
            "%d %b %Y",           # 1 Jan 2000
            "%m/%d/%Y",           # 01/01/2000
            "%d/%m/%Y",           # 01/01/2000 (European)
            "%Y/%m/%d",           # 2000/01/01
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def execute(
        self,
        operation: str,
        date: str = None,
        date2: str = None,
        days: int = None,
        timezone: str = None
    ) -> str:
        """
        Execute datetime operation.
        
        Args:
            operation: The operation to perform.
            date: Primary date for operations.
            date2: Secondary date for difference calculation.
            days: Number of days to add/subtract.
            timezone: Timezone name.
            
        Returns:
            Result as string.
        """
        try:
            if operation == "now":
                return self._get_now(timezone)
            
            elif operation == "day_of_week":
                if not date:
                    return "Error: 'date' parameter required for day_of_week"
                return self._get_day_of_week(date)
            
            elif operation == "add_days":
                if not date:
                    return "Error: 'date' parameter required for add_days"
                if days is None:
                    return "Error: 'days' parameter required for add_days"
                return self._add_days(date, days)
            
            elif operation == "difference":
                if not date or not date2:
                    return "Error: 'date' and 'date2' parameters required for difference"
                return self._get_difference(date, date2)
            
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_now(self, timezone: str = None) -> str:
        """Get current datetime."""
        now = datetime.now()
        
        # Try to handle timezone if specified
        if timezone:
            try:
                import zoneinfo
                tz = zoneinfo.ZoneInfo(timezone)
                now = datetime.now(tz)
                return now.strftime(f"%Y-%m-%d %H:%M:%S ({timezone})")
            except Exception:
                # Fallback if timezone not available
                pass
        
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_day_of_week(self, date_str: str) -> str:
        """Get day of week for a date."""
        dt = self._parse_date(date_str)
        if not dt:
            return f"Error: Could not parse date '{date_str}'"
        day_name = calendar.day_name[dt.weekday()]
        formatted = dt.strftime("%Y-%m-%d")
        return f"{formatted} is a {day_name}"
    
    def _add_days(self, date_str: str, days: int) -> str:
        """Add or subtract days from a date."""
        dt = self._parse_date(date_str)
        if not dt:
            return f"Error: Could not parse date '{date_str}'"
        new_dt = dt + timedelta(days=days)
        return new_dt.strftime("%Y-%m-%d")
    
    def _get_difference(self, date1_str: str, date2_str: str) -> str:
        """Calculate difference between two dates."""
        dt1 = self._parse_date(date1_str)
        dt2 = self._parse_date(date2_str)
        if not dt1:
            return f"Error: Could not parse date '{date1_str}'"
        if not dt2:
            return f"Error: Could not parse date '{date2_str}'"
        diff = abs((dt2 - dt1).days)
        return f"{diff} days between {dt1.strftime('%Y-%m-%d')} and {dt2.strftime('%Y-%m-%d')}"
