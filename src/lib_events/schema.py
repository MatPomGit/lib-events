import json
import math
from typing import Any

class ValidationError(ValueError):
    """Report a schema subset validation failure."""


def _is_type(value: Any, expected: str) -> bool:
    """Return whether a value matches one JSON-Schema primitive type."""
    checks = {
        "null": lambda item: item is None,
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int)
        and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "string": lambda item: isinstance(item, str),
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
    }
    try:
        return checks[expected](value)
    except KeyError as exc:
        raise ValidationError(
            f"Unsupported schema type in validator: {expected}"
        ) from exc


def validate_schema_subset(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> None:
    """Validate the supported JSON-Schema subset."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{path}: non-finite JSON number is invalid")

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = (
            expected_type if isinstance(expected_type, list) else [expected_type]
        )
        if not any(_is_type(value, item) for item in allowed):
            raise ValidationError(
                f"{path}: expected type {allowed}, got {type(value).__name__}"
            )

    if "const" in schema and value != schema["const"]:
        raise ValidationError(
            f"{path}: expected constant {schema['const']!r}, got {value!r}"
        )

    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(
            f"{path}: value {value!r} is not in allowed enum {schema['enum']!r}"
        )

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(f"{path}: string is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValidationError(f"{path}: string is longer than maxLength")

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and "minimum" in schema
        and value < schema["minimum"]
    ):
        raise ValidationError(
            f"{path}: {value} is below minimum {schema['minimum']}"
        )

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValidationError(f"{path}: missing required fields {missing}")

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValidationError(
                    f"{path}: unknown fields are not allowed: {unknown}"
                )

        for key, item in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                validate_schema_subset(
                    item,
                    child_schema,
                    f"{path}.{key}",
                )

    if isinstance(value, list):
        if schema.get("uniqueItems") and len(
            {json.dumps(item, sort_keys=True) for item in value}
        ) != len(value):
            raise ValidationError(f"{path}: array items must be unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                validate_schema_subset(
                    item,
                    item_schema,
                    f"{path}[{index}]",
                )


