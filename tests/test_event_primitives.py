import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lib_events.schema import validate_schema_subset, ValidationError
from lib_events.queue import validate_delivery, validate_event_identity_consistency, TelemetryQueueError

class EventTests(unittest.TestCase):
    def test_subset(self):
        s = {"type":"object","required":["x"],"properties":{"x":{"type":"number"}},"additionalProperties":False}
        validate_schema_subset({"x":2},s)
        for value in ({"x":float("nan")},{"x":True},{"x":2,"y":1},{}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                validate_schema_subset(value,s)
    def test_delivery(self):
        validate_delivery({"delivery_state":"pending","attempt_count":0})
        for value in ({"delivery_state":"in_flight","attempt_count":0},{"delivery_state":"blocked","attempt_count":1},{"delivery_state":"pending","attempt_count":True}):
            with self.assertRaises(TelemetryQueueError):
                validate_delivery(value)
    def test_identity(self):
        a={"event":{"event_id":"synthetic","value":1}}
        validate_event_identity_consistency([a,a])
        with self.assertRaises(TelemetryQueueError):
            validate_event_identity_consistency([a,{"event":{"event_id":"synthetic","value":2}}])
