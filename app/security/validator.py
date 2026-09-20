"""
Argument validation engine for Vector Desktop AI Assistant.
Validates tool parameters against JSON schemas before execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationResult:
    """
    Result returned by argument validation checks.
    """
    valid: bool
    cleaned_args: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ArgumentValidator:
    """
    Validates tool arguments against parameter specifications to ensure type safety.
    """

    @staticmethod
    def validate(parameters_schema: Dict[str, Any], provided_args: Dict[str, Any]) -> ValidationResult:
        """
        Validates provided arguments against a tool's parameters schema.
        Parameters schema follows JSON Schema format:
        {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100, "description": "..."}
            },
            "required": ["level"]
        }
        """
        errors: List[str] = []
        cleaned_args: Dict[str, Any] = {}
        properties = parameters_schema.get("properties", {})
        required = parameters_schema.get("required", [])

        # Check required fields
        for req_field in required:
            if req_field not in provided_args or provided_args[req_field] is None:
                errors.append(f"Missing required parameter: '{req_field}'")

        # Validate provided parameters
        for arg_name, arg_val in provided_args.items():
            if arg_name not in properties:
                # Ignore unexpected arguments or flag if strict
                continue

            param_spec = properties[arg_name]
            expected_type = param_spec.get("type")

            # Type checking
            if expected_type == "integer":
                if not isinstance(arg_val, int) or isinstance(arg_val, bool):
                    try:
                        arg_val = int(arg_val)
                    except (ValueError, TypeError):
                        errors.append(f"Parameter '{arg_name}' must be an integer, got {type(arg_val).__name__}")
                        continue
                
                # Check numeric bounds
                min_val = param_spec.get("minimum")
                max_val = param_spec.get("maximum")
                if min_val is not None and arg_val < min_val:
                    errors.append(f"Parameter '{arg_name}' value {arg_val} is below minimum ({min_val})")
                if max_val is not None and arg_val > max_val:
                    errors.append(f"Parameter '{arg_name}' value {arg_val} exceeds maximum ({max_val})")

            elif expected_type == "number":
                if not isinstance(arg_val, (int, float)) or isinstance(arg_val, bool):
                    try:
                        arg_val = float(arg_val)
                    except (ValueError, TypeError):
                        errors.append(f"Parameter '{arg_name}' must be a number, got {type(arg_val).__name__}")
                        continue

            elif expected_type == "string":
                if not isinstance(arg_val, str):
                    arg_val = str(arg_val)

            elif expected_type == "boolean":
                if not isinstance(arg_val, bool):
                    if isinstance(arg_val, str):
                        arg_val = arg_val.lower() in ("true", "1", "yes")
                    else:
                        arg_val = bool(arg_val)

            cleaned_args[arg_name] = arg_val

        return ValidationResult(
            valid=len(errors) == 0,
            cleaned_args=cleaned_args,
            errors=errors
        )
