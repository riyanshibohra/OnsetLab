"""
OnsetLab Validator
==================
Validates generated training data against MCP tool schemas.

Ensures:
1. Tool names exist in the schema
2. Parameters match expected types
3. Required parameters are present
4. JSON is valid
5. No placeholders or template variables

Filters out bad examples and reports quality score.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

from .schemas import ToolSchema


@dataclass
class ValidationError:
    """Represents a single validation error."""
    line_number: int
    error_type: str
    message: str
    example_preview: str = ""


@dataclass
class ValidationResult:
    """Result of validating a training data file."""
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    errors: list[ValidationError] = field(default_factory=list)
    
    # Error counts by type
    error_counts: dict = field(default_factory=dict)
    
    @property
    def quality_score(self) -> float:
        """Returns quality score as percentage (0-100)."""
        if self.total_examples == 0:
            return 0.0
        return (self.valid_examples / self.total_examples) * 100
    
    @property
    def is_good(self) -> bool:
        """Returns True if quality score is above 90%."""
        return self.quality_score >= 90
    
    def add_error(self, error: ValidationError):
        """Add an error and update counts."""
        self.errors.append(error)
        self.error_counts[error.error_type] = self.error_counts.get(error.error_type, 0) + 1


class Validator:
    """
    Validates training data against MCP tool schemas.
    
    Usage:
        >>> from onsetlab.utils import Validator, ToolSchema
        >>> 
        >>> tools = [ToolSchema(...), ToolSchema(...)]
        >>> validator = Validator(tools=tools)
        >>> result = validator.validate("training_data.jsonl")
        >>> print(f"Quality: {result.quality_score}%")
    """
    
    # Common placeholders that indicate incomplete generation
    # NOTE: Be careful not to match valid JSON structures!
    PLACEHOLDER_PATTERNS = [
        r'\{\{[^{}]+\}\}',        # {{date}}, {{time}} - but not nested JSON
        r'<[A-Z][A-Z_]{2,}>',     # <TODAY>, <USER_NAME> - all caps, 3+ chars
        r'\[[A-Z][A-Z_]*\]',      # [NAME], [DATE] - all caps in brackets (not JSON arrays)
        r'<[a-z_]+_placeholder>', # <any_placeholder>
        r'INSERT_\w+_HERE',       # INSERT_VALUE_HERE patterns
        r'YOUR_\w+_HERE',         # YOUR_EMAIL_HERE patterns
        r'PLACEHOLDER',           # literal PLACEHOLDER
    ]
    
    def __init__(
        self,
        tools: Union[list[ToolSchema], list[dict]] = None,
        tools_path: str = None
    ):
        """
        Initialize validator with tool schemas.
        
        Args:
            tools: List of ToolSchema objects or tool dicts
            tools_path: Path to JSON file with tool schemas
        """
        if tools:
            # Convert to dict format for validation
            self.tools = {}
            for t in tools:
                if isinstance(t, ToolSchema):
                    self.tools[t.name] = t.to_dict()
                else:
                    self.tools[t["name"]] = t
        elif tools_path:
            with open(tools_path) as f:
                tools_data = json.load(f)
            self.tools = {t["name"]: t for t in tools_data}
        else:
            raise ValueError("Must provide either 'tools' or 'tools_path'")
        
        self.tool_names = set(self.tools.keys())
        
        # Build normalized name mapping for fuzzy matching
        self._name_map = {}
        for name in self.tool_names:
            # Map normalized versions to actual name
            normalized = self._normalize_name(name)
            self._name_map[normalized] = name
            self._name_map[name] = name  # Also keep original
    
    def _normalize_name(self, name: str) -> str:
        """Normalize tool name for matching (handles underscores, hyphens, case)."""
        return name.lower().replace("-", "_").replace(" ", "_")
    
    def _find_tool(self, name: str) -> Optional[str]:
        """Find actual tool name from possibly-varied input."""
        if name in self.tool_names:
            return name
        normalized = self._normalize_name(name)
        return self._name_map.get(normalized)
    
    def _extract_tool_call(self, assistant_content: str) -> Optional[dict]:
        """
        Extract tool call from assistant message.
        
        Returns None if no valid tool call found.
        """
        # Look for <tool_call>...</tool_call>
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', assistant_content, re.DOTALL)
        if not match:
            return None
        
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    
    def _has_placeholders(self, text: str) -> bool:
        """Check if text contains placeholder patterns."""
        for pattern in self.PLACEHOLDER_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
    
    def _validate_parameter_type(self, value, expected_type: str) -> bool:
        """Validate that a value matches expected JSON schema type."""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)),
            "integer": lambda v: isinstance(v, int),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }
        
        check = type_checks.get(expected_type)
        if check:
            return check(value)
        return True  # Unknown types pass
    
    def _validate_tool_call(self, tool_call: dict, line_num: int) -> list[ValidationError]:
        """Validate a single tool call against schemas."""
        errors = []
        
        # Check tool name exists
        tool_name = tool_call.get("tool")
        if not tool_name:
            errors.append(ValidationError(
                line_number=line_num,
                error_type="missing_tool_name",
                message="Tool call missing 'tool' field"
            ))
            return errors
        
        # Try fuzzy matching for tool name
        actual_tool_name = self._find_tool(tool_name)
        if actual_tool_name is None:
            # Show closest matches for debugging
            close_matches = [n for n in self.tool_names if self._normalize_name(tool_name)[:5] in self._normalize_name(n)]
            hint = f" (close: {close_matches[:3]})" if close_matches else ""
            errors.append(ValidationError(
                line_number=line_num,
                error_type="unknown_tool",
                message=f"Unknown tool: '{tool_name}'{hint}"
            ))
            return errors
        
        # Get tool schema using actual (normalized) name
        tool_schema = self.tools[actual_tool_name]
        input_schema = tool_schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        # Get parameters from tool call
        params = tool_call.get("parameters", {})
        
        # Check required parameters
        for req_param in required:
            if req_param not in params:
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="missing_required_param",
                    message=f"Tool '{tool_name}' missing required param: '{req_param}'"
                ))
        
        # Check parameter types and unknown params
        for param_name, param_value in params.items():
            if param_name not in properties:
                # Unknown parameter - could be valid depending on schema
                # Only warn if schema has additionalProperties: false
                if input_schema.get("additionalProperties") == False:
                    errors.append(ValidationError(
                        line_number=line_num,
                        error_type="unknown_param",
                        message=f"Tool '{tool_name}' has unknown param: '{param_name}'"
                    ))
            else:
                # Validate type
                expected_type = properties[param_name].get("type")
                if expected_type and not self._validate_parameter_type(param_value, expected_type):
                    errors.append(ValidationError(
                        line_number=line_num,
                        error_type="type_mismatch",
                        message=f"Param '{param_name}' expected {expected_type}, got {type(param_value).__name__}"
                    ))
            
            # Check for placeholders in string values
            if isinstance(param_value, str) and self._has_placeholders(param_value):
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="placeholder_found",
                    message=f"Param '{param_name}' contains placeholder: '{param_value}'"
                ))
        
        return errors
    
    def _validate_example(self, example: dict, line_num: int) -> list[ValidationError]:
        """Validate a single training example."""
        errors = []
        
        # Check structure
        if "messages" not in example:
            errors.append(ValidationError(
                line_number=line_num,
                error_type="invalid_structure",
                message="Example missing 'messages' field"
            ))
            return errors
        
        messages = example["messages"]
        
        # Find assistant messages with tool calls
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            
            content = msg.get("content", "")
            
            # Check for placeholders in content
            if self._has_placeholders(content):
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="placeholder_found",
                    message=f"Assistant message contains placeholder",
                    example_preview=content[:100]
                ))
            
            # If it's a tool call, validate it
            if "<tool_call>" in content:
                tool_call = self._extract_tool_call(content)
                if tool_call is None:
                    errors.append(ValidationError(
                        line_number=line_num,
                        error_type="invalid_json",
                        message="Could not parse tool call JSON",
                        example_preview=content[:100]
                    ))
                else:
                    errors.extend(self._validate_tool_call(tool_call, line_num))
        
        return errors
    
    def validate(self, data_path: str) -> ValidationResult:
        """
        Validate a training data file.
        
        Args:
            data_path: Path to JSONL file
            
        Returns:
            ValidationResult with stats and errors
        """
        result = ValidationResult()
        
        with open(data_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                result.total_examples += 1
                
                # Parse JSON
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    result.add_error(ValidationError(
                        line_number=line_num,
                        error_type="json_parse_error",
                        message=f"Invalid JSON: {e}",
                        example_preview=line[:100]
                    ))
                    result.invalid_examples += 1
                    continue
                
                # Validate example
                errors = self._validate_example(example, line_num)
                
                if errors:
                    for error in errors:
                        result.add_error(error)
                    result.invalid_examples += 1
                else:
                    result.valid_examples += 1
        
        return result
    
    def validate_and_filter(self, data_path: str, output_path: str = None) -> tuple[ValidationResult, str]:
        """
        Validate and create a filtered file with only valid examples.
        
        Args:
            data_path: Path to input JSONL file
            output_path: Path for filtered output (default: adds _valid suffix)
            
        Returns:
            (ValidationResult, output_path)
        """
        if output_path is None:
            path = Path(data_path)
            output_path = str(path.parent / f"{path.stem}_valid{path.suffix}")
        
        result = ValidationResult()
        valid_examples = []
        
        with open(data_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                result.total_examples += 1
                
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    result.add_error(ValidationError(
                        line_number=line_num,
                        error_type="json_parse_error",
                        message=f"Invalid JSON: {e}"
                    ))
                    result.invalid_examples += 1
                    continue
                
                errors = self._validate_example(example, line_num)
                
                if errors:
                    for error in errors:
                        result.add_error(error)
                    result.invalid_examples += 1
                else:
                    result.valid_examples += 1
                    valid_examples.append(example)
        
        # Write valid examples
        with open(output_path, 'w') as f:
            for example in valid_examples:
                f.write(json.dumps(example) + "\n")
        
        return result, output_path
    
    def print_report(self, result: ValidationResult):
        """Print a formatted validation report."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)
        
        # Summary
        print(f"\n📈 Summary:")
        print(f"   Total examples:   {result.total_examples}")
        print(f"   Valid examples:   {result.valid_examples} ✅")
        print(f"   Invalid examples: {result.invalid_examples} ❌")
        print(f"   Quality score:    {result.quality_score:.1f}%")
        
        # Quality indicator
        if result.quality_score >= 95:
            print(f"\n   🌟 Excellent quality!")
        elif result.quality_score >= 90:
            print(f"\n   ✅ Good quality - ready for training")
        elif result.quality_score >= 80:
            print(f"\n   ⚠️ Acceptable - consider regenerating some examples")
        else:
            print(f"\n   ❌ Poor quality - regenerate training data")
        
        # Error breakdown
        if result.error_counts:
            print(f"\n🔍 Error breakdown:")
            for error_type, count in sorted(result.error_counts.items(), key=lambda x: -x[1]):
                print(f"   {error_type}: {count}")
        
        # Sample errors (first 5)
        if result.errors:
            print(f"\n📋 Sample errors (showing first 5):")
            for error in result.errors[:5]:
                print(f"\n   Line {error.line_number}: [{error.error_type}]")
                print(f"   {error.message}")
                if error.example_preview:
                    print(f"   Preview: {error.example_preview[:80]}...")
        
        print("\n" + "=" * 60)


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_training_data(
    data_path: str,
    tools: Union[list[ToolSchema], list[dict], str]
) -> ValidationResult:
    """
    Validate training data against tool schemas.
    
    Args:
        data_path: Path to JSONL file
        tools: List of ToolSchema objects, tool dicts, or path to JSON file
        
    Returns:
        ValidationResult with stats and errors
    """
    if isinstance(tools, str):
        validator = Validator(tools_path=tools)
    else:
        validator = Validator(tools=tools)
    
    return validator.validate(data_path)
