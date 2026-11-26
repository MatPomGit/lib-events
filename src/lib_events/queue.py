import json
from typing import Any

class TelemetryQueueError(ValueError):
    """Report an invalid telemetry queue record."""


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


def validate_delivery(record: dict[str, Any]) -> None:
    """Validate delivery metadata independently of the event payload."""
    if not isinstance(record, dict):
        raise TelemetryQueueError("record must be an object")
    state = record.get("delivery_state")
    if state not in ("pending", "in_flight", "acknowledged", "blocked"):
        raise TelemetryQueueError("Unsupported delivery_state.")
    attempt_count = _require_nonnegative_int(record, "attempt_count")
    last_error = record.get("last_transport_error")
    if last_error is not None and (not isinstance(last_error, str) or not last_error.strip()):
        raise TelemetryQueueError("Invalid transport error.")
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
            allow_nan=False,
        )
        previous = seen.get(event_id)
        if previous is not None and previous != canonical:
            raise TelemetryQueueError(
                f"Integrity conflict for event_id {event_id!r}."
            )
        seen[event_id] = canonical


