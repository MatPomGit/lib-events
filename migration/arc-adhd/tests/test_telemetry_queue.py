"""Test ARC-ADHD durable telemetry queue semantics."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.validate_telemetry_queue import (
    TelemetryQueueError,
    load_json,
    validate_event_identity_consistency,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[1]
PENDING = ROOT / "examples" / "telemetry_queue.pending.json"
ACKNOWLEDGED = ROOT / "examples" / "telemetry_queue.acknowledged.json"


class TelemetryQueueValidationTests(unittest.TestCase):
    """Verify queue states, privacy, and idempotent event identity."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load committed synthetic queue records."""
        cls.pending = load_json(PENDING)
        cls.acknowledged = load_json(ACKNOWLEDGED)

    def test_reference_records_are_valid(self) -> None:
        """Accept pending and acknowledged synthetic examples."""
        validate_record(self.pending)
        validate_record(self.acknowledged)

    def test_pending_record_cannot_be_acknowledged(self) -> None:
        """Reject acknowledgement metadata on a pending record."""
        record = copy.deepcopy(self.pending)
        record["acknowledgement_id"] = "ACK-INVALID"
        with self.assertRaises(TelemetryQueueError):
            validate_record(record)

    def test_acknowledged_record_requires_ack_id(self) -> None:
        """Reject acknowledged state without explicit acknowledgement ID."""
        record = copy.deepcopy(self.acknowledged)
        record["acknowledgement_id"] = None
        with self.assertRaises(TelemetryQueueError):
            validate_record(record)

    def test_direct_identity_key_in_payload_is_rejected(self) -> None:
        """Reject direct participant identity fields in research payload."""
        record = copy.deepcopy(self.pending)
        record["event"]["payload"]["email"] = "synthetic@example.invalid"
        with self.assertRaises(TelemetryQueueError):
            validate_record(record)

    def test_precise_location_key_is_rejected(self) -> None:
        """Reject precise location keys in research payload."""
        record = copy.deepcopy(self.pending)
        record["event"]["payload"]["latitude"] = 50.0
        with self.assertRaises(TelemetryQueueError):
            validate_record(record)

    def test_same_event_id_with_same_payload_is_idempotent(self) -> None:
        """Allow identical event retries under the same event ID."""
        first = copy.deepcopy(self.pending)
        second = copy.deepcopy(self.pending)
        second["delivery_state"] = "in_flight"
        second["attempt_count"] = 1
        second["last_attempt_wallclock"] = "2026-09-28T00:00:00Z"
        validate_record(second)
        validate_event_identity_consistency([first, second])

    def test_same_event_id_with_different_payload_is_conflict(self) -> None:
        """Reject reused event IDs with different scientific content."""
        first = copy.deepcopy(self.pending)
        second = copy.deepcopy(self.pending)
        second["event"]["payload"]["proposed_action"] = "increase_cognitive"
        with self.assertRaises(TelemetryQueueError):
            validate_event_identity_consistency([first, second])


if __name__ == "__main__":
    unittest.main()
