# API

`validate_schema_subset(value, schema)` checks a limited JSON Schema subset, not full Draft 2020-12. `validate_delivery(record)` checks delivery metadata. `validate_event_identity_consistency(records)` rejects conflicting content for repeated event IDs. Exceptions: `ValidationError`, `TelemetryQueueError`. No network I/O. See [contract](../contracts/api.json).
