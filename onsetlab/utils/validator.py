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
6. Multi-turn format is correct (role sequences)
7. Tool results are valid JSON
8. Final message is a summary (not hanging tool call)

Filters out bad examples and reports quality score.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Union, List
from pathlib import Path

from .schemas import ToolSchema


@dataclass
class ValidationError:
    """Represents a single validation error."""
    line_number: int
    error_type: str
    message: str
    severity: str = "error"  # "error" or "warning"
    example_preview: str = ""


@dataclass
class ValidationResult:
    """Result of validating a training data file."""
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    warnings_count: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    
    # Error counts by type
    error_counts: dict = field(default_factory=dict)
    
    # Statistics
    single_tool_count: int = 0
    multi_tool_count: int = 0
    edge_case_count: int = 0
    
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
        if error.severity == "warning":
            self.warnings_count += 1


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
    
    def _validate_example(self, example: dict, line_num: int) -> tuple[List[ValidationError], str]:
        """
        Validate a single training example.
        
        Returns:
            (errors, example_type) where example_type is 'single_tool', 'multi_tool', or 'edge_case'
        """
        errors = []
        example_type = "edge_case"  # Default
        
        # Check structure
        if "messages" not in example:
            errors.append(ValidationError(
                line_number=line_num,
                error_type="invalid_structure",
                message="Example missing 'messages' field"
            ))
            return errors, example_type
        
        messages = example["messages"]
        
        # Check for empty messages
        if not messages:
            errors.append(ValidationError(
                line_number=line_num,
                error_type="empty_messages",
                message="Messages array is empty"
            ))
            return errors, example_type
        
        # Extract roles for sequence validation
        roles = [m.get("role") for m in messages]
        
        # Check minimum structure: should have at least system, user, assistant
        if len(messages) < 3:
            errors.append(ValidationError(
                line_number=line_num,
                error_type="insufficient_messages",
                message=f"Example has only {len(messages)} messages (need at least 3: system, user, assistant)",
                severity="warning"
            ))
        
        # Validate role sequence
        role_errors = self._validate_role_sequence(roles, line_num)
        errors.extend(role_errors)
        
        # Count tool calls and tool results to determine type
        tool_call_count = 0
        tool_result_count = roles.count("tool")
        
        # Validate each message
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Check for empty content
            if not content and role != "tool":  # Tool results can be empty in edge cases
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="empty_content",
                    message=f"Message {i} ({role}) has empty content",
                    severity="warning"
                ))
            
            # Validate based on role
            if role == "assistant":
                # Check for placeholders
                if self._has_placeholders(content):
                    errors.append(ValidationError(
                        line_number=line_num,
                        error_type="placeholder_found",
                        message=f"Assistant message contains placeholder",
                        example_preview=content[:100]
                    ))
                
                # If it's a tool call, validate it
                if "<tool_call>" in content:
                    tool_call_count += 1
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
            
            elif role == "tool":
                # Validate tool result is valid JSON
                tool_errors = self._validate_tool_result(content, line_num, i)
                errors.extend(tool_errors)
            
            elif role == "user":
                # Check for placeholders in user message
                if self._has_placeholders(content):
                    errors.append(ValidationError(
                        line_number=line_num,
                        error_type="placeholder_found",
                        message=f"User message contains placeholder",
                        example_preview=content[:100]
                    ))
        
        # Determine example type
        if tool_result_count > 0:
            example_type = "multi_tool"
        elif tool_call_count > 0:
            example_type = "single_tool"
        else:
            example_type = "edge_case"
        
        # For multi-turn, check that it ends with a summary (not a hanging tool call)
        if tool_result_count > 0:
            last_assistant = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    last_assistant = msg.get("content", "")
                    break
            
            if last_assistant and "<tool_call>" in last_assistant:
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="hanging_tool_call",
                    message="Multi-turn example ends with tool call instead of summary",
                    severity="warning"
                ))
        
        return errors, example_type
    
    def _validate_role_sequence(self, roles: List[str], line_num: int) -> List[ValidationError]:
        """Validate that role sequence is valid."""
        errors = []
        
        # Valid roles
        valid_roles = {"system", "user", "assistant", "tool"}
        
        for i, role in enumerate(roles):
            if role not in valid_roles:
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="invalid_role",
                    message=f"Unknown role: '{role}'"
                ))
        
        # Check sequence rules
        for i in range(1, len(roles)):
            prev_role = roles[i - 1]
            curr_role = roles[i]
            
            # Tool result must follow assistant (tool call)
            if curr_role == "tool" and prev_role != "assistant":
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="invalid_sequence",
                    message=f"'tool' role must follow 'assistant', but follows '{prev_role}'"
                ))
            
            # After tool result, must be assistant
            if prev_role == "tool" and curr_role != "assistant":
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="invalid_sequence",
                    message=f"After 'tool' must come 'assistant', but got '{curr_role}'"
                ))
        
        return errors
    
    def _validate_tool_result(self, content: str, line_num: int, msg_index: int) -> List[ValidationError]:
        """Validate a tool result message."""
        errors = []
        
        if not content:
            errors.append(ValidationError(
                line_number=line_num,
                error_type="empty_tool_result",
                message=f"Tool result at message {msg_index} is empty",
                severity="warning"
            ))
            return errors
        
        # Try to parse as JSON (tool results should be JSON)
        try:
            json.loads(content)
        except json.JSONDecodeError:
            # Not JSON - could be plain text error message, which is OK
            # Only warn if it looks like it should be JSON
            if content.strip().startswith("{") or content.strip().startswith("["):
                errors.append(ValidationError(
                    line_number=line_num,
                    error_type="invalid_tool_result_json",
                    message=f"Tool result looks like JSON but failed to parse",
                    example_preview=content[:100],
                    severity="warning"
                ))
        
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
                errors, example_type = self._validate_example(example, line_num)
                
                # Count by type
                if example_type == "single_tool":
                    result.single_tool_count += 1
                elif example_type == "multi_tool":
                    result.multi_tool_count += 1
                else:
                    result.edge_case_count += 1
                
                # Only count as invalid if there are actual errors (not just warnings)
                actual_errors = [e for e in errors if e.severity == "error"]
                
                if actual_errors:
                    for error in errors:
                        result.add_error(error)
                    result.invalid_examples += 1
                else:
                    # Add warnings but still count as valid
                    for error in errors:
                        if error.severity == "warning":
                            result.add_error(error)
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
                
                errors, example_type = self._validate_example(example, line_num)
                
                # Count by type
                if example_type == "single_tool":
                    result.single_tool_count += 1
                elif example_type == "multi_tool":
                    result.multi_tool_count += 1
                else:
                    result.edge_case_count += 1
                
                # Only count as invalid if there are actual errors (not just warnings)
                actual_errors = [e for e in errors if e.severity == "error"]
                
                if actual_errors:
                    for error in errors:
                        result.add_error(error)
                    result.invalid_examples += 1
                else:
                    # Valid - add to output
                    for error in errors:
                        if error.severity == "warning":
                            result.add_error(error)
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
        if result.warnings_count > 0:
            print(f"   Warnings:         {result.warnings_count} ⚠️")
        print(f"   Quality score:    {result.quality_score:.1f}%")
        
        # Example type breakdown
        print(f"\n📁 Example Types:")
        print(f"   Single-tool:  {result.single_tool_count}")
        print(f"   Multi-tool:   {result.multi_tool_count}")
        print(f"   Edge cases:   {result.edge_case_count}")
        
        # Quality indicator
        if result.quality_score >= 95:
            print(f"\n   🌟 Excellent quality!")
        elif result.quality_score >= 90:
            print(f"\n   ✅ Good quality - ready for training")
        elif result.quality_score >= 80:
            print(f"\n   ⚠️ Acceptable - consider regenerating some examples")
        else:
            print(f"\n   ❌ Poor quality - regenerate training data")
        
        # Error breakdown (separate errors from warnings)
        if result.error_counts:
            # Separate errors and warnings
            errors_only = {k: v for k, v in result.error_counts.items() 
                          if not any(e.error_type == k and e.severity == "warning" for e in result.errors)}
            warnings_only = {k: v for k, v in result.error_counts.items() 
                           if any(e.error_type == k and e.severity == "warning" for e in result.errors)}
            
            if errors_only:
                print(f"\n❌ Errors:")
                for error_type, count in sorted(errors_only.items(), key=lambda x: -x[1]):
                    print(f"   {error_type}: {count}")
            
            if warnings_only:
                print(f"\n⚠️ Warnings:")
                for error_type, count in sorted(warnings_only.items(), key=lambda x: -x[1]):
                    print(f"   {error_type}: {count}")
        
        # Sample errors (first 5 actual errors, then warnings)
        actual_errors = [e for e in result.errors if e.severity == "error"]
        warnings = [e for e in result.errors if e.severity == "warning"]
        
        if actual_errors:
            print(f"\n📋 Sample errors (showing first 5):")
            for error in actual_errors[:5]:
                print(f"\n   Line {error.line_number}: [{error.error_type}]")
                print(f"   {error.message}")
                if error.example_preview:
                    print(f"   Preview: {error.example_preview[:80]}...")
        
        if warnings and len(actual_errors) < 3:
            print(f"\n📋 Sample warnings (showing first 3):")
            for warning in warnings[:3]:
                print(f"\n   Line {warning.line_number}: [{warning.error_type}]")
                print(f"   {warning.message}")
        
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
