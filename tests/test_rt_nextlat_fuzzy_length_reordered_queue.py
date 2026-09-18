"""Standard-library checks for safe observation of the revised training order."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import rt_nextlat_fuzzy_length_reordered_queue as module


class ReorderedObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name) / "observer"
        self.training = Path(self.temporary.name) / "training"
        self.runtime.mkdir()
        self.training.mkdir()
        # Bypass only the repository-relative constructor for isolated fixtures;
        # the production inherited run/retention methods are not executed here.
        self.queue = module.ReorderedLengthQueue.__new__(module.ReorderedLengthQueue)
        self.queue.runtime = self.runtime
        self.queue.config = {"training_queue": str(self.training),
                             "expected_training_order": ["value", "head", "input"]}

    def packet(self, phase="complete", order=("value", "head", "input"), endpoint=15000,
               arm_status="complete"):
        result = {"phase": phase, "pid": os.getpid(), "current_variant": "value",
                  "completed_variants": [{"variant": name, "endpoint": endpoint, "status": arm_status}
                                         for name in order]}
        (self.training / "status.json").write_text(json.dumps(result))
        return result

    def status(self):
        return json.loads((self.runtime / "status.json").read_text())

    def test_accepts_only_matched_reordered_completion(self):
        self.packet()
        self.assertTrue(self.queue.wait_for_training())
        for kwargs in ({"order": ("input", "value", "head")}, {"endpoint": 14999},
                       {"endpoint": 15001}, {"arm_status": "stopped"},
                       {"order": ("value", "head")}, {"order": ("value", "head", "input", "input")}):
            with self.subTest(kwargs=kwargs):
                self.packet(**kwargs)
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    self.queue.wait_for_training()

    def test_stop_precedes_even_completed_training(self):
        self.packet()
        (self.runtime / "STOP").touch()
        self.assertFalse(self.queue.wait_for_training())
        self.assertEqual(self.status()["phase"], "cancelled_before_evaluation")

    def test_failed_or_stopped_training_never_evaluates(self):
        for phase in ("failed", "stopped_and_retained", "stopped_before_next_variant",
                      "cancelled_before_variants", "stopped_after_preflight",
                      "cancelled_before_launch", "cancelled_before_training"):
            with self.subTest(phase=phase):
                self.packet(phase=phase)
                self.assertFalse(self.queue.wait_for_training())
                self.assertEqual(self.status()["phase"], "blocked_training_incomplete")
                self.assertEqual(self.status()["training_phase"], phase)

    def test_wait_then_exact_completion(self):
        self.packet(phase="training_value")
        with patch.object(module.time, "sleep", side_effect=lambda _: self.packet()) as sleep:
            self.assertTrue(self.queue.wait_for_training())
            sleep.assert_called_once_with(30)
        self.assertEqual(self.status()["phase"], "waiting_for_training_queue")
        self.assertEqual(self.status()["expected_training_order"], ["value", "head", "input"])

    def test_wait_observes_cancellation_before_next_read(self):
        self.packet(phase="training_value")
        with patch.object(module.time, "sleep", side_effect=lambda _: (self.runtime / "STOP").touch()):
            self.assertFalse(self.queue.wait_for_training())
        self.assertEqual(self.status()["phase"], "cancelled_before_evaluation")

    def test_dead_training_supervisor_fails(self):
        self.packet(phase="training_value")
        with patch.object(module.os, "kill", side_effect=ProcessLookupError):
            with self.assertRaisesRegex(RuntimeError, "supervisor exited"):
                self.queue.wait_for_training()

    def test_missing_or_unapproved_expected_order_fails(self):
        for expected in (None, ["input", "value", "head"], ["value", "head"]):
            with self.subTest(expected=expected):
                self.queue.config["expected_training_order"] = expected
                with self.assertRaisesRegex(RuntimeError, "requires value/head/input"):
                    self.queue.wait_for_training()


if __name__ == "__main__":
    unittest.main()
