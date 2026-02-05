"""DateTime tool for date and time operations."""

from datetime import datetime, timedelta
from typing import Any, Dict
import calendar

from .base import BaseTool


class DateTime(BaseTool):
    """
    DateTime tool for getting current time and date calculations.
    """
    
    name = "DateTime"
    description = "Get current date/time, calculate date differences, or find day of week for any date."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation: 'now' (current datetime), 'day_of_week' (for a date), 'add_days' (add/subtract days), 'difference' (days between dates)",
                    "enum": ["now", "day_of_week", "add_days", "difference"]
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (for day_of_week, add_days)"
                },
                "date2": {
                    "type": "string",
                    "description": "Second date in YYYY-MM-DD format (for difference)"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to add (negative to subtract)"
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g., 'UTC', 'US/Eastern'). Default: local time."
                }
            },
            "required": ["operation"]
        }
    
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
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_name = calendar.day_name[dt.weekday()]
        return f"{date_str} is a {day_name}"
    
    def _add_days(self, date_str: str, days: int) -> str:
        """Add or subtract days from a date."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        new_dt = dt + timedelta(days=days)
        return new_dt.strftime("%Y-%m-%d")
    
    def _get_difference(self, date1_str: str, date2_str: str) -> str:
        """Calculate difference between two dates."""
        dt1 = datetime.strptime(date1_str, "%Y-%m-%d")
        dt2 = datetime.strptime(date2_str, "%Y-%m-%d")
        diff = abs((dt2 - dt1).days)
        return f"{diff} days between {date1_str} and {date2_str}"
