"""RandomGenerator tool for generating random values."""

import random
import string
import uuid
import secrets
from typing import Any, Dict
from .base import BaseTool


class RandomGenerator(BaseTool):
    """
    Generate random values: numbers, UUIDs, passwords, choices, dice rolls.
    """
    
    name = "RandomGenerator"
    description = "Generate random: numbers (integer/float/range), UUIDs, secure passwords, pick from choices, dice rolls, coin flips."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "What to generate",
                    "enum": [
                        "integer", "float", "uuid", "password",
                        "choice", "shuffle", "dice", "coin", "sample"
                    ]
                },
                "min": {
                    "type": "number",
                    "description": "Minimum value (for integer/float)"
                },
                "max": {
                    "type": "number",
                    "description": "Maximum value (for integer/float)"
                },
                "length": {
                    "type": "integer",
                    "description": "Length (for password)"
                },
                "choices": {
                    "type": "string",
                    "description": "Comma-separated options (for choice/shuffle/sample)"
                },
                "count": {
                    "type": "integer",
                    "description": "How many to generate (for dice, sample, integer)"
                },
                "sides": {
                    "type": "integer",
                    "description": "Number of sides (for dice, default 6)"
                }
            },
            "required": ["operation"]
        }
    
    def execute(
        self,
        operation: str,
        min: float = None,
        max: float = None,
        length: int = None,
        choices: str = None,
        count: int = None,
        sides: int = None
    ) -> str:
        """Execute random generation operation."""
        operation = operation.lower().strip()
        
        try:
            if operation == "integer":
                return self._random_integer(min, max, count)
            
            elif operation == "float":
                return self._random_float(min, max)
            
            elif operation == "uuid":
                return self._generate_uuid()
            
            elif operation == "password":
                return self._generate_password(length)
            
            elif operation == "choice":
                if not choices:
                    return "Error: 'choices' required (comma-separated options)"
                return self._random_choice(choices)
            
            elif operation == "shuffle":
                if not choices:
                    return "Error: 'choices' required (comma-separated items)"
                return self._shuffle(choices)
            
            elif operation == "sample":
                if not choices:
                    return "Error: 'choices' required (comma-separated items)"
                return self._sample(choices, count or 1)
            
            elif operation == "dice":
                return self._roll_dice(sides or 6, count or 1)
            
            elif operation == "coin":
                return self._flip_coin(count or 1)
            
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _random_integer(self, min_val: float, max_val: float, count: int = None) -> str:
        """Generate random integer(s)."""
        min_int = int(min_val) if min_val is not None else 1
        max_int = int(max_val) if max_val is not None else 100
        
        if min_int > max_int:
            min_int, max_int = max_int, min_int
        
        if count and count > 1:
            numbers = [random.randint(min_int, max_int) for _ in range(min(count, 100))]
            return ", ".join(str(n) for n in numbers)
        else:
            return str(random.randint(min_int, max_int))
    
    def _random_float(self, min_val: float, max_val: float) -> str:
        """Generate random float."""
        min_f = float(min_val) if min_val is not None else 0.0
        max_f = float(max_val) if max_val is not None else 1.0
        
        if min_f > max_f:
            min_f, max_f = max_f, min_f
        
        result = random.uniform(min_f, max_f)
        return f"{result:.4f}"
    
    def _generate_uuid(self) -> str:
        """Generate a UUID."""
        return str(uuid.uuid4())
    
    def _generate_password(self, length: int = None) -> str:
        """Generate a secure password."""
        length = length or 16
        length = max(8, min(length, 128))  # Clamp between 8-128
        
        # Ensure at least one of each required character type
        password = [
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.digits),
            secrets.choice("!@#$%^&*()_+-=[]{}|;:,.<>?")
        ]
        
        # Fill the rest randomly
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
        password += [secrets.choice(all_chars) for _ in range(length - 4)]
        
        # Shuffle to randomize position of required chars
        random.shuffle(password)
        
        return "".join(password)
    
    def _random_choice(self, choices: str) -> str:
        """Pick randomly from comma-separated choices."""
        options = [c.strip() for c in choices.split(",") if c.strip()]
        if not options:
            return "Error: No valid choices provided"
        return random.choice(options)
    
    def _shuffle(self, choices: str) -> str:
        """Shuffle comma-separated items."""
        items = [c.strip() for c in choices.split(",") if c.strip()]
        if not items:
            return "Error: No valid items provided"
        random.shuffle(items)
        return ", ".join(items)
    
    def _sample(self, choices: str, count: int) -> str:
        """Pick N random items without replacement."""
        items = [c.strip() for c in choices.split(",") if c.strip()]
        if not items:
            return "Error: No valid items provided"
        
        count = min(count, len(items))
        selected = random.sample(items, count)
        return ", ".join(selected)
    
    def _roll_dice(self, sides: int, count: int) -> str:
        """Roll dice."""
        sides = max(2, min(sides, 100))  # 2-100 sides
        count = max(1, min(count, 20))   # 1-20 dice
        
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)
        
        if count == 1:
            return f"Rolled d{sides}: {rolls[0]}"
        else:
            rolls_str = ", ".join(str(r) for r in rolls)
            return f"Rolled {count}d{sides}: {rolls_str} (total: {total})"
    
    def _flip_coin(self, count: int) -> str:
        """Flip coin(s)."""
        count = max(1, min(count, 20))
        
        flips = [random.choice(["Heads", "Tails"]) for _ in range(count)]
        
        if count == 1:
            return flips[0]
        else:
            heads = flips.count("Heads")
            tails = flips.count("Tails")
            flips_str = ", ".join(flips)
            return f"{flips_str} ({heads} heads, {tails} tails)"
