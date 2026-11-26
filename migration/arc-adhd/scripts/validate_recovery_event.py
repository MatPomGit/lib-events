#!/usr/bin/env python3
"""Validate ARC-ADHD recovery-event JSON records.

This validator intentionally uses only the Python standard library. It checks
the JSON-Schema subset used by schemas/recovery_event.schema.json and then
applies ARC-ADHD semantic invariants that are easier to express explicitly in
code.

Example:
    python scripts/validate_recovery_event.py examples/recovery_event.observed.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schemas" / "recovery_event.schema.json"

RECOVERY_MODEL_WEIGHTS = {
    "B": {
        "balanced_rule_success": 1.0,
    },
    "B+S": {
        "balanced_rule_success": 0.5,
        "correct_go_rt_slowing": 0.5,
    },
    "B+V": {
        "balanced_rule_success": 0.5,
        "correct_go_rt_variability_increase": 0.5,
    },
    "B+S+V": {
        "balanced_rule_success": 0.5,
        "correct_go_rt_slowing": 0.25,
        "correct_go_rt_variability_increase": 0.25,
    },
}


class ValidationError(ValueError):
    """Report a recovery-event validation failure."""


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


def _validate_schema_subset(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> None:
    """Validate the JSON-Schema subset used by the recovery-event schema."""
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
                _validate_schema_subset(
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
                _validate_schema_subset(
                    item,
                    item_schema,
                    f"{path}[{index}]",
                )


def _validate_semantics(event: dict[str, Any]) -> None:
    """Validate ARC-ADHD recovery-event semantic invariants."""
    observed = event["recovery_observed"]
    censored = event["right_censored"]
    telemetry_complete = event["telemetry_complete"]
    data_valid = event["recovery_data_valid"]
    context_unchanged = event["recovery_measurement_context_unchanged"]
    invalid_reason = event["invalid_reason"]
    monitoring_onset = event.get("recovery_monitoring_onset_ms")
    threshold_entry = event.get("recovery_threshold_entry_ms")
    confirmation = event.get("recovery_confirmation_ms")
    pre_monitoring_latency = event.get("pre_monitoring_latency_ms")
    latency = event.get("recovery_latency_ms")
    total_return_latency = event.get("total_return_latency_ms")
    censor_reason = event.get("censor_reason")
    params = event["recovery_parameters"]
    model_id = event["recovery_feature_model_id"]
    score_q = event["recovery_score_q"]
    score_components = event["recovery_score_components"]
    challenge_eligibility = event["challenge_eligibility"]
    timing_eligible = challenge_eligibility["timing_feature_eligible"]
    timing_reasons = challenge_eligibility["timing_unavailability_reasons"]
    recovery_reference = event["target_rule_recovery_reference"]
    measurement_context = event["recovery_measurement_context"]

    if challenge_eligibility["core_eligible"] is not True:
        raise ValidationError(
            "$.challenge_eligibility.core_eligible: a recovery event can "
            "exist only for a core-eligible challenge"
        )

    if recovery_reference["source_state"] != "RULE_CALIBRATION":
        raise ValidationError(
            "$.target_rule_recovery_reference.source_state: must be RULE_CALIBRATION"
        )
    if recovery_reference["rule_id"] != measurement_context["task_rule_id"]:
        raise ValidationError(
            "$.target_rule_recovery_reference.rule_id: must match recovery target rule"
        )
    if (
        recovery_reference["go_stimulus_id"]
        != measurement_context["go_stimulus_id"]
    ):
        raise ValidationError(
            "$.target_rule_recovery_reference.go_stimulus_id: must match "
            "recovery GO-stimulus identity"
        )
    if (
        recovery_reference["go_successes"] > recovery_reference["go_total"]
        or recovery_reference["nogo_successes"] > recovery_reference["nogo_total"]
    ):
        raise ValidationError(
            "$.target_rule_recovery_reference: successes cannot exceed totals"
        )
    if (
        recovery_reference["go_total"] + recovery_reference["nogo_total"]
        > recovery_reference["valid_trials"]
    ):
        raise ValidationError(
            "$.target_rule_recovery_reference: GO/NO-GO totals cannot exceed "
            "valid_trials"
        )

    if timing_eligible:
        if timing_reasons:
            raise ValidationError(
                "$.challenge_eligibility.timing_unavailability_reasons: "
                "timing-eligible event requires an empty reason list"
            )
        if recovery_reference["correct_go_rt_trials"] < 4:
            raise ValidationError(
                "$.target_rule_recovery_reference.correct_go_rt_trials: "
                "timing-eligible Phase 0 event requires at least 4"
            )
        if not isinstance(recovery_reference["timing_profile_id"], str):
            raise ValidationError(
                "$.target_rule_recovery_reference.timing_profile_id: "
                "timing-eligible event requires a profile ID"
            )
    else:
        if not timing_reasons:
            raise ValidationError(
                "$.challenge_eligibility.timing_unavailability_reasons: "
                "timing-ineligible event requires at least one reason"
            )
        if model_id != "B":
            raise ValidationError(
                "$.recovery_feature_model_id: timing-ineligible events can "
                "only use the behavior-only B model"
            )

    if model_id != "B" and not timing_eligible:
        raise ValidationError(
            "$.recovery_feature_model_id: timing-enriched model requires "
            "timing_feature_eligible=true"
        )

    if model_id not in RECOVERY_MODEL_WEIGHTS:
        raise ValidationError(
            "$.recovery_feature_model_id: unknown recovery feature model"
        )

    if data_valid:
        if not isinstance(score_q, (int, float)) or isinstance(score_q, bool):
            raise ValidationError(
                "$.recovery_score_q: valid recovery data require a numeric q"
            )
        if score_q > 0:
            raise ValidationError(
                "$.recovery_score_q: non-compensatory q cannot exceed zero"
            )
        if not isinstance(score_components, dict) or not score_components:
            raise ValidationError(
                "$.recovery_score_components: valid recovery data require "
                "non-empty component deficits"
            )

        expected_weights = RECOVERY_MODEL_WEIGHTS[model_id]
        if set(score_components) != set(expected_weights):
            raise ValidationError(
                "$.recovery_score_components: component keys must match the "
                "selected recovery feature model exactly"
            )

        recomputed_q = 0.0
        for feature, weight in expected_weights.items():
            deficit = score_components[feature]
            if (
                not isinstance(deficit, (int, float))
                or isinstance(deficit, bool)
                or deficit < 0
            ):
                raise ValidationError(
                    "$.recovery_score_components: deficits must be "
                    "non-negative numbers"
                )
            recomputed_q -= float(weight) * float(deficit)

        if abs(float(score_q) - recomputed_q) > 1e-9:
            raise ValidationError(
                "$.recovery_score_q: q does not match the selected model "
                "weights and stored component deficits"
            )
    else:
        if score_q is not None:
            raise ValidationError(
                "$.recovery_score_q: invalid recovery data require null q"
            )
        if score_components is not None:
            raise ValidationError(
                "$.recovery_score_components: invalid recovery data require "
                "null components"
            )

    if observed and censored:
        raise ValidationError(
            "$: recovery_observed and right_censored cannot both be true"
        )

    if data_valid:
        if not telemetry_complete:
            raise ValidationError(
                "$: recovery_data_valid requires complete telemetry"
            )
        if not context_unchanged:
            raise ValidationError(
                "$: recovery_data_valid requires an unchanged measurement context"
            )
        if observed == censored:
            raise ValidationError(
                "$: valid recovery data require exactly one of "
                "recovery_observed or right_censored to be true"
            )
        if invalid_reason is not None:
            raise ValidationError(
                "$.invalid_reason: valid recovery data require null invalid_reason"
            )
    else:
        if observed or censored:
            raise ValidationError(
                "$: invalid recovery data cannot be recorded as observed or censored"
            )
        for field_name, value in (
            ("recovery_threshold_entry_ms", threshold_entry),
            ("recovery_confirmation_ms", confirmation),
            ("recovery_latency_ms", latency),
            ("total_return_latency_ms", total_return_latency),
        ):
            if value is not None:
                raise ValidationError(
                    f"$.{field_name}: invalid recovery data require null"
                )
        if censor_reason is not None:
            raise ValidationError(
                "$.censor_reason: invalid recovery data are not right-censored"
            )
        if invalid_reason is None:
            raise ValidationError(
                "$.invalid_reason: invalid recovery data require a reason"
            )

    if not context_unchanged:
        if data_valid:
            raise ValidationError(
                "$: a changed recovery measurement context invalidates the event"
            )
        if invalid_reason != "measurement_context_changed":
            raise ValidationError(
                "$.invalid_reason: changed measurement context requires "
                "measurement_context_changed"
            )

    challenge_onset = event["challenge_onset_ms"]

    if monitoring_onset is None:
        if pre_monitoring_latency is not None:
            raise ValidationError(
                "$.pre_monitoring_latency_ms: null monitoring onset requires "
                "null pre-monitoring latency"
            )
    else:
        if not isinstance(monitoring_onset, int) or isinstance(monitoring_onset, bool):
            raise ValidationError(
                "$.recovery_monitoring_onset_ms: must be integer or null"
            )
        if monitoring_onset < challenge_onset:
            raise ValidationError(
                "$.recovery_monitoring_onset_ms: cannot precede challenge onset"
            )
        expected_pre = monitoring_onset - challenge_onset
        if pre_monitoring_latency != expected_pre:
            raise ValidationError(
                "$.pre_monitoring_latency_ms: must equal monitoring onset "
                "minus challenge onset"
            )

    if data_valid and monitoring_onset is None:
        raise ValidationError(
            "$.recovery_monitoring_onset_ms: valid recovery data require "
            "entry into RECOVERY_MONITORING"
        )

    if observed:
        for field_name, value in (
            ("recovery_threshold_entry_ms", threshold_entry),
            ("recovery_confirmation_ms", confirmation),
            ("recovery_latency_ms", latency),
            ("total_return_latency_ms", total_return_latency),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValidationError(
                    f"$.{field_name}: observed recovery requires an integer"
                )

        if threshold_entry < monitoring_onset:
            raise ValidationError(
                "$.recovery_threshold_entry_ms: cannot precede monitoring onset"
            )
        if confirmation < threshold_entry:
            raise ValidationError(
                "$.recovery_confirmation_ms: cannot precede threshold entry"
            )
        if confirmation > monitoring_onset + params["censor_limit_ms"]:
            raise ValidationError(
                "$.recovery_confirmation_ms: stable recovery must be "
                "confirmed within the censor window"
            )

        expected_recovery = threshold_entry - monitoring_onset
        expected_total = threshold_entry - challenge_onset
        if latency != expected_recovery:
            raise ValidationError(
                "$.recovery_latency_ms: must equal threshold entry minus "
                "recovery-monitoring onset"
            )
        if total_return_latency != expected_total:
            raise ValidationError(
                "$.total_return_latency_ms: must equal threshold entry minus "
                "challenge onset"
            )
        if total_return_latency != pre_monitoring_latency + latency:
            raise ValidationError(
                "$.total_return_latency_ms: must equal pre-monitoring latency "
                "plus recovery latency"
            )
        if latency > params["censor_limit_ms"]:
            raise ValidationError(
                "$.recovery_latency_ms: observed recovery cannot exceed "
                "the censor limit"
            )
        if censor_reason is not None:
            raise ValidationError(
                "$.censor_reason: observed recovery cannot have a censor reason"
            )
        if float(score_q) < -float(params["delta"]):
            raise ValidationError(
                "$.recovery_score_q: observed recovery must close at or "
                "above the configured -delta threshold"
            )

    if censored:
        for field_name, value in (
            ("recovery_threshold_entry_ms", threshold_entry),
            ("recovery_confirmation_ms", confirmation),
            ("recovery_latency_ms", latency),
            ("total_return_latency_ms", total_return_latency),
        ):
            if value is not None:
                raise ValidationError(
                    f"$.{field_name}: censored recovery requires null"
                )
        if censor_reason is None:
            raise ValidationError(
                "$.censor_reason: censored recovery requires a reason"
            )

    if not telemetry_complete:
        if not event.get("missingness_codes"):
            raise ValidationError(
                "$.missingness_codes: incomplete telemetry requires at least "
                "one missingness code"
            )
        if data_valid:
            raise ValidationError(
                "$: incomplete telemetry cannot produce valid recovery data"
            )

    window = event.get("reference_window")
    if window is not None:
        start = window["start_ms"]
        end = window["end_ms"]
        onset = event["challenge_onset_ms"]
        if start >= end:
            raise ValidationError(
                "$.reference_window: start_ms must be less than end_ms"
            )
        if end > onset:
            raise ValidationError(
                "$.reference_window: reference window must end no later "
                "than challenge onset"
            )


def validate_event(
    event: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Validate one recovery event against structure and ARC semantics."""
    _validate_schema_subset(event, schema)
    _validate_semantics(event)


def load_json(path: Path) -> Any:
    """Load and return strict UTF-8 JSON without non-finite numbers."""
    def reject_non_finite(token: str) -> float:
        raise ValidationError(f"{path}: non-finite JSON number {token}")

    def parse_finite_float(token: str) -> float:
        number = float(token)
        if not math.isfinite(number):
            raise ValidationError(f"{path}: non-finite JSON number {token}")
        return number

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_non_finite,
            parse_float=parse_finite_float,
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def main() -> int:
    """Run command-line validation and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Validate ARC-ADHD recovery-event JSON records."
    )
    parser.add_argument(
        "events",
        nargs="+",
        type=Path,
        help="Recovery-event JSON files to validate.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Schema path. Defaults to schemas/recovery_event.schema.json.",
    )
    args = parser.parse_args()

    schema = load_json(args.schema)
    failures = 0

    for path in args.events:
        try:
            event = load_json(path)
            if not isinstance(event, dict):
                raise ValidationError(f"{path}: top-level JSON must be an object")
            validate_event(event, schema)
        except (OSError, ValidationError) as exc:
            failures += 1
            print(f"INVALID {path}: {exc}")
        else:
            print(f"VALID {path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
