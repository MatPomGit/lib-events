#!/usr/bin/env python3
"""Validate ARC-ADHD telemetry queue records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schemas" / "telemetry_queue_record.schema.json"

FORBIDDEN_IDENTITY_KEYS = {
    "name",
    "full_name",
    "first_name",
    "last_name",
    "caregiver_name",
    "email",
    "email_address",
    "phone",
    "phone_number",
    "postal_address",
    "address",
    "ip_address",
    "latitude",
    "longitude",
    "gps",
    "precise_location",
}

TRANSPORT_ERRORS = {
    None,
    "network_unavailable",
    "timeout",
    "server_unavailable",
    "authentication_failure",
    "payload_rejected",
    "schema_mismatch",
    "integrity_conflict",
    "local_storage_failure",
    "unknown_transport_error",
}


class TelemetryQueueError(ValueError):
    """Report an invalid telemetry queue record."""


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON from a path."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TelemetryQueueError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _require_string(
    mapping: dict[str, Any],
    key: str,
    *,
    allow_null: bool = False,
) -> str | None:
    """Return one required string or allowed null value."""
    value = mapping.get(key)
    if allow_null and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TelemetryQueueError(f"{key} must be a non-empty string.")
    return value


def _require_nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    """Return one required non-negative integer."""
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TelemetryQueueError(
            f"{key} must be a non-negative integer."
        )
    return value


def _find_forbidden_keys(
    value: Any,
    path: str = "$",
) -> list[str]:
    """Return forbidden direct-identity keys found recursively."""
    findings: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            child_path = f"{path}.{key}"
            if normalized in FORBIDDEN_IDENTITY_KEYS:
                findings.append(child_path)
            findings.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(
                _find_forbidden_keys(child, f"{path}[{index}]")
            )

    return findings


def validate_record(record: dict[str, Any]) -> None:
    """Validate telemetry queue structure, delivery state, and privacy rules."""
    if record.get("queue_schema_version") != "0.1":
        raise TelemetryQueueError(
            "queue_schema_version must equal '0.1'."
        )

    state = record.get("delivery_state")
    if state not in {"pending", "in_flight", "acknowledged", "blocked"}:
        raise TelemetryQueueError("Unsupported delivery_state.")

    attempt_count = _require_nonnegative_int(record, "attempt_count")
    last_error = record.get("last_transport_error")
    if last_error not in TRANSPORT_ERRORS:
        raise TelemetryQueueError("Unsupported last_transport_error.")

    event = record.get("event")
    if not isinstance(event, dict):
        raise TelemetryQueueError("event must be an object.")

    for key in (
        "event_schema_version",
        "event_id",
        "participant_pseudonym",
        "session_id",
        "event_type",
        "intervention_version",
        "configuration_hash",
    ):
        _require_string(event, key)

    _require_nonnegative_int(event, "sequence_number")
    _require_nonnegative_int(event, "monotonic_ms")

    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise TelemetryQueueError("event.payload must be an object.")

    forbidden = _find_forbidden_keys(payload, "$.event.payload")
    if forbidden:
        raise TelemetryQueueError(
            "Direct identity/location keys are forbidden in research "
            f"payload: {forbidden}"
        )

    acknowledgement_id = record.get("acknowledgement_id")
    acknowledged_at = record.get("acknowledged_wallclock")

    if state == "pending":
        if acknowledgement_id is not None or acknowledged_at is not None:
            raise TelemetryQueueError(
                "Pending records cannot contain acknowledgement metadata."
            )

    if state == "in_flight":
        if attempt_count < 1:
            raise TelemetryQueueError(
                "In-flight records require attempt_count >= 1."
            )
        if acknowledgement_id is not None or acknowledged_at is not None:
            raise TelemetryQueueError(
                "In-flight records cannot already be acknowledged."
            )

    if state == "acknowledged":
        _require_string(record, "acknowledgement_id")
        _require_string(record, "acknowledged_wallclock")
        if attempt_count < 1:
            raise TelemetryQueueError(
                "Acknowledged records require attempt_count >= 1."
            )
        if last_error is not None:
            raise TelemetryQueueError(
                "Acknowledged record cannot retain an active transport error."
            )

    if state == "blocked":
        if last_error is None:
            raise TelemetryQueueError(
                "Blocked record requires a transport/integrity error."
            )
        if acknowledgement_id is not None or acknowledged_at is not None:
            raise TelemetryQueueError(
                "Blocked records cannot be acknowledged."
            )


def validate_event_identity_consistency(
    records: list[dict[str, Any]],
) -> None:
    """Reject the same event ID appearing with different scientific content."""
    seen: dict[str, str] = {}

    for record in records:
        event = record["event"]
        event_id = str(event["event_id"])
        canonical = json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
        )
        previous = seen.get(event_id)
        if previous is not None and previous != canonical:
            raise TelemetryQueueError(
                f"Integrity conflict for event_id {event_id!r}."
            )
        seen[event_id] = canonical


def main() -> int:
    """Validate one or more telemetry queue record files."""
    parser = argparse.ArgumentParser(
        description="Validate ARC-ADHD telemetry queue records."
    )
    parser.add_argument(
        "records",
        nargs="+",
        type=Path,
        help="Telemetry queue record JSON files.",
    )
    args = parser.parse_args()

    loaded: list[dict[str, Any]] = []
    failures = 0

    for path in args.records:
        try:
            record = load_json(path)
            if not isinstance(record, dict):
                raise TelemetryQueueError(
                    f"{path}: top-level JSON must be an object."
                )
            validate_record(record)
            loaded.append(record)
        except (OSError, TelemetryQueueError) as exc:
            failures += 1
            print(f"INVALID {path}: {exc}")
        else:
            print(f"VALID {path}")

    if not failures:
        try:
            validate_event_identity_consistency(loaded)
        except TelemetryQueueError as exc:
            failures += 1
            print(f"INVALID event identity set: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
