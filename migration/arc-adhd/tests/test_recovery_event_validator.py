"""Test ARC-ADHD recovery-event validation invariants."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_recovery_event import (
    DEFAULT_SCHEMA,
    ValidationError,
    load_json as load_strict_json,
    validate_event,
)

ROOT = Path(__file__).resolve().parents[1]
OBSERVED_EXAMPLE = ROOT / "examples" / "recovery_event.observed.json"
CENSORED_EXAMPLE = ROOT / "examples" / "recovery_event.censored.json"
INVALID_CONTEXT_EXAMPLE = (
    ROOT / "examples" / "recovery_event.invalid_context.json"
)


def load_json(path: Path) -> dict:
    """Load a test JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


class RecoveryEventValidationTests(unittest.TestCase):
    """Verify structural and semantic recovery-event rules."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the shared schema once for the test class."""
        cls.schema = load_json(DEFAULT_SCHEMA)
        cls.observed = load_json(OBSERVED_EXAMPLE)
        cls.censored = load_json(CENSORED_EXAMPLE)
        cls.invalid_context = load_json(INVALID_CONTEXT_EXAMPLE)

    def test_observed_example_is_valid(self) -> None:
        """Accept the synthetic observed-recovery example."""
        validate_event(self.observed, self.schema)

    def test_censored_example_is_valid(self) -> None:
        """Accept the synthetic right-censored example."""
        validate_event(self.censored, self.schema)

    def test_context_changed_example_is_valid_record(self) -> None:
        """Accept a complete record whose recovery measurement is invalid."""
        validate_event(self.invalid_context, self.schema)

    def test_observed_and_censored_cannot_both_be_true(self) -> None:
        """Reject an event marked as both recovered and censored."""
        event = copy.deepcopy(self.observed)
        event["right_censored"] = True
        event["censor_reason"] = "event_window_end"

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_incomplete_telemetry_requires_missingness_code(self) -> None:
        """Reject incomplete telemetry without an explicit missingness reason."""
        event = copy.deepcopy(self.observed)
        event["telemetry_complete"] = False
        event["missingness_codes"] = []

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_incomplete_telemetry_cannot_remain_scientifically_valid(self) -> None:
        """Reject valid recovery status when required telemetry is incomplete."""
        event = copy.deepcopy(self.observed)
        event["telemetry_complete"] = False
        event["missingness_codes"] = ["software_error"]

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_observed_latency_cannot_exceed_censor_limit(self) -> None:
        """Reject recovery recorded after the event censor limit."""
        event = copy.deepcopy(self.observed)
        event["recovery_latency_ms"] = (
            event["recovery_parameters"]["censor_limit_ms"] + 1
        )

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_recovery_latency_is_anchored_to_monitoring_onset(self) -> None:
        """Reject latency measured from the wrong time origin."""
        event = copy.deepcopy(self.observed)
        event["recovery_latency_ms"] += 1000

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_total_return_equals_pre_plus_recovery(self) -> None:
        """Reject a total return time inconsistent with its decomposition."""
        event = copy.deepcopy(self.observed)
        event["total_return_latency_ms"] += 1

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_confirmation_must_occur_within_censor_window(self) -> None:
        """Require the complete stable window to be observed before censoring."""
        event = copy.deepcopy(self.observed)
        event["recovery_confirmation_ms"] = (
            event["recovery_monitoring_onset_ms"]
            + event["recovery_parameters"]["censor_limit_ms"]
            + 1
        )

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_recovery_monitoring_cannot_precede_challenge(self) -> None:
        """Reject impossible recovery-monitoring time ordering."""
        event = copy.deepcopy(self.observed)
        event["recovery_monitoring_onset_ms"] = (
            event["challenge_onset_ms"] - 1
        )

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_reference_window_cannot_extend_past_challenge(self) -> None:
        """Reject a reference window that includes post-challenge time."""
        event = copy.deepcopy(self.observed)
        event["reference_window"]["end_ms"] = event["challenge_onset_ms"] + 1

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_valid_event_requires_observed_or_censored_state(self) -> None:
        """Reject valid recovery data with no terminal recovery outcome."""
        event = copy.deepcopy(self.observed)
        event["recovery_observed"] = False
        event["recovery_latency_ms"] = None
        event["right_censored"] = False

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_context_change_invalidates_recovery_measurement(self) -> None:
        """Reject a recovery result if its measurement context changed."""
        event = copy.deepcopy(self.observed)
        event["recovery_measurement_context_unchanged"] = False
        event["invalid_reason"] = "measurement_context_changed"

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_invalid_recovery_data_are_not_right_censored(self) -> None:
        """Reject conflation of scientific invalidity with right-censoring."""
        event = copy.deepcopy(self.invalid_context)
        event["right_censored"] = True
        event["censor_reason"] = "event_window_end"

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_q_must_match_selected_model_weights(self) -> None:
        """Reject a q value inconsistent with stored B/S/V deficits."""
        event = copy.deepcopy(self.observed)
        event["recovery_score_q"] = -0.9

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_component_set_must_match_model_id(self) -> None:
        """Reject extra or missing components for the selected model."""
        event = copy.deepcopy(self.observed)
        del event["recovery_score_components"][
            "correct_go_rt_variability_increase"
        ]

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)


    def test_nonfinite_recovery_scores_are_rejected_at_schema_boundary(self) -> None:
        """Reject NaN and infinities before arithmetic or recovery inference."""
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=repr(value)):
                event = copy.deepcopy(self.observed)
                event["recovery_score_q"] = value
                with self.assertRaisesRegex(ValidationError, "non-finite"):
                    validate_event(event, self.schema)

    def test_json_input_rejects_nonfinite_numbers(self) -> None:
        """Reject non-JSON constants and overflowing numeric exponents."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.json"
            for token in ("NaN", "Infinity", "-Infinity", "1e999"):
                with self.subTest(token=token):
                    path.write_text(
                        '{"recovery_score_q": ' + token + '}',
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValidationError, "non-finite"):
                        load_strict_json(path)

    def test_positive_q_is_rejected(self) -> None:
        """Reject positive compensation in recovery-event records."""
        event = copy.deepcopy(self.observed)
        event["recovery_score_q"] = 0.1

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_timing_model_requires_timing_eligible_event(self) -> None:
        """Reject timing-enriched q when event timing was not eligible."""
        event = copy.deepcopy(self.observed)
        event["challenge_eligibility"]["timing_feature_eligible"] = False
        event["challenge_eligibility"]["timing_unavailability_reasons"] = [
            "timing_profile_unqualified"
        ]
        event["target_rule_recovery_reference"]["correct_go_rt_trials"] = 0
        event["target_rule_recovery_reference"]["timing_profile_id"] = None

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_behavior_only_model_remains_valid_without_timing(self) -> None:
        """Allow a core behavioral recovery event on timing-unqualified BYOD."""
        event = copy.deepcopy(self.observed)
        event["challenge_eligibility"]["timing_feature_eligible"] = False
        event["challenge_eligibility"]["timing_unavailability_reasons"] = [
            "timing_profile_unqualified"
        ]
        event["target_rule_recovery_reference"]["correct_go_rt_trials"] = 0
        event["target_rule_recovery_reference"]["timing_profile_id"] = None
        event["recovery_feature_model_id"] = "B"
        event["recovery_score_components"] = {
            "balanced_rule_success": 0.2
        }
        event["recovery_score_q"] = -0.2
        validate_event(event, self.schema)

    def test_target_rule_reference_must_match_recovery_rule(self) -> None:
        """Reject B/S/V reference collected under a different target rule."""
        event = copy.deepcopy(self.observed)
        event["target_rule_recovery_reference"]["rule_id"] = "A"

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_target_rule_reference_go_stimulus_must_match_context(self) -> None:
        """Reject B/S/V reference for a different GO stimulus identity."""
        event = copy.deepcopy(self.observed)
        event["target_rule_recovery_reference"]["go_stimulus_id"] = "blue"

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_target_reference_behavior_counts_are_valid(self) -> None:
        """Reject impossible behavior counts in the q reference."""
        event = copy.deepcopy(self.observed)
        event["target_rule_recovery_reference"]["go_successes"] = 7

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_timing_eligible_event_requires_minimum_rt_reference(self) -> None:
        """Reject timing-enriched event without the four-RT Phase 0 reference."""
        event = copy.deepcopy(self.observed)
        event["target_rule_recovery_reference"]["correct_go_rt_trials"] = 3

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_timing_eligible_event_requires_empty_unavailability_reasons(self) -> None:
        """Reject contradictory timing-eligibility metadata."""
        event = copy.deepcopy(self.observed)
        event["challenge_eligibility"]["timing_unavailability_reasons"] = [
            "timing_profile_unqualified"
        ]

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)

    def test_observed_recovery_must_close_above_threshold(self) -> None:
        """Reject an observed recovery record whose final q is below -delta."""
        event = copy.deepcopy(self.observed)
        event["recovery_score_components"] = {
            "balanced_rule_success": 1.0,
            "correct_go_rt_slowing": 1.0,
            "correct_go_rt_variability_increase": 1.0,
        }
        event["recovery_score_q"] = -1.0

        with self.assertRaises(ValidationError):
            validate_event(event, self.schema)


if __name__ == "__main__":
    unittest.main()
