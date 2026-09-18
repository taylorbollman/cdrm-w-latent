"""Five focused CPU-only report guards: comparison, selection, provenance and curves."""
import copy
import hashlib
import unittest
from unittest.mock import patch

from scripts import rt_a5_window_report as report


def contracts(position="sinusoidal", window=None):
    baseline = {"schema": "rt-a5-nextlat-training-v1", "source_sha256": "0" * 64,
                "objective": {"latent_weight": 1.0}, "optimizer": "unchanged-adam",
                "data_manifest_sha256": "1" * 64, "seed": 1234,
                "model_config": {"alibi": True, "rope": False, "n_layers": 2}}
    candidate = copy.deepcopy(baseline)
    candidate.update(schema=report.TRAIN_SCHEMA, source_sha256="2" * 64, **report.NEW_CONTRACT_FIELDS)
    candidate["model_config"]["alibi"] = position == "alibi"
    details = {"kind": "alibi", "alibi": True, "rope": False, "learned_position_parameters": False}
    if position == "sinusoidal":
        details.update(kind="fixed_sinusoidal", alibi=False, base=10000.0, amplitude=1.0,
                       position_origin=0, token_embedding_scale=1.0, persistent_position_buffers=False,
                       nextlat_conditioning="raw next-operation token embedding without positional addition")
    candidate["experiment_config"] = {
        "schema": "rt-a5-window-config-v1", "architecture": "rt", "width": 512,
        "position_encoding": position, "second_layer_window": window,
        "initialization": {"kind": "original_mitchell", "identity_centered_overrides": False,
                           "learned_tensor_changes": 0}, "position_details": details,
        "attention": {"first_layer": "full causal recurrent attention",
                      "second_layer": "full causal recurrent attention" if window is None else
                                      "self provisional K/V and immediately previous permanent output K/V",
                      "second_layer_allowed_keys": "0 <= key <= query" if window is None else
                                                   "max(0, query - 1) <= key <= query",
                      "first_token": "self provisional K/V only", "recurrent_write_rho": 1.0,
                      "gradient_truncation": False,
                      "implementation": "original recurrent block" if window is None else
                                        "parameter-free block subclass supplies additive mask to original forward/backward"}}
    return baseline, candidate


def metric(step, role, accuracy=.5):
    n, length = 102400, {"dev": 12, "ood_dev": 36}[role]
    return {"update": step, "role": role, "rows": n, "length": length, "tokens": n*length,
            "isolated_state_accuracy": [accuracy]*length, "cumulative_prefix_exactness": [accuracy]*length,
            "per_position_ce": [1.0]*length, "token_accuracy": accuracy, "whole_word_exact_match": accuracy,
            "final_state_accuracy": accuracy, "ce": 1.0}


def selection_fixture(sinusoidal_accuracy=.5):
    criterion = {"checkpoint": 10000, "role": "ood_dev", "rows": 102400,
                 "exact_tie": "alibi", "primary": "mean cumulative-prefix exactness E(t) over t=13..36 inclusive"}
    selection = {"schema": "rt-a5-window-position-selection-v1", "status": "selected",
                 "criterion": criterion, "confirmation_evaluated": False,
                 "matched_order_chain": "a" * 64, "arms": {}}
    arms = {}
    for position, name, accuracy in [("alibi", report.ARMS[0], .5),
                                    ("sinusoidal", report.ARMS[1], sinusoidal_accuracy)]:
        relative = f".runtime/fixture/{position}"
        # Selection records are created on the host; reports can run inside Docker.
        host_directory = "/home/taylorbollman/cdrm-w-latent/" + relative
        actual_directory = report.ROOT / relative
        checkpoint = {"path": str(actual_directory / "step-010000.pt"),
                      "sha256": "b" * 64, "bytes": 123}
        recorded_checkpoint = {**checkpoint, "path": "/workspace/cdrm-w-latent/" + relative + "/step-010000.pt",
                               "completed_updates": 10000, "examples_seen": 10240000}
        metrics = {role: metric(10000, role, accuracy) for role in report.ROLES}
        arms[name] = {"metrics": {"10000": metrics}, "checkpoints": {"10000": checkpoint},
                      "input_files": {"report": {"path": str(actual_directory / "report.json"), "sha256": "c" * 64}},
                      "report": {"order_chain": "a" * 64}}
        count_sum = round(accuracy * 102400) * 24
        selection["arms"][position] = {"directory": host_directory, "report_sha256": "c" * 64,
                                       "checkpoint": recorded_checkpoint, "endpoint_metrics": metrics,
                                       "exact_prefix_count_sum_13_36": count_sum,
                                       "denominator": 102400 * 24, "score": count_sum / (102400 * 24)}
    winner = "sinusoidal" if sinusoidal_accuracy > .5 else "alibi"
    selection.update(selected_position=winner,
                     selected_full_attention_directory=selection["arms"][winner]["directory"],
                     next_run={"position_encoding": winner, "second_layer_window": 2,
                               "initialization": "fresh original Mitchell", "updates": 10000})
    return selection, arms, {"position_selection": criterion}


class WindowReportChecks(unittest.TestCase):
    def test_only_positions_and_second_layer_window_contract_changes_allowed(self):
        for position in ("alibi", "sinusoidal"):
            for window in (None, 2):
                baseline, candidate = contracts(position, window)
                report.compare_contracts(baseline, candidate)
                changed = copy.deepcopy(candidate)
                changed["optimizer"] = "different"
                with self.assertRaisesRegex(ValueError, "Unplanned"):
                    report.compare_contracts(baseline, changed)
        baseline, candidate = contracts("sinusoidal", 2)
        for field, value in (("first_layer", "window2"), ("gradient_truncation", True)):
            changed = copy.deepcopy(candidate)
            changed["experiment_config"]["attention"][field] = value
            with self.assertRaisesRegex(ValueError, "semantics"):
                report.compare_contracts(baseline, changed)
        candidate["experiment_config"]["position_details"]["learned_position_parameters"] = True
        with self.assertRaisesRegex(ValueError, "learned"):
            report.compare_contracts(baseline, candidate)

    def test_selection_reproduces_integer_scores_tie_rule_and_evidence_identity(self):
        for accuracy, expected in ((.5, "alibi"), (.75, "sinusoidal"), (.25, "alibi")):
            selection, arms, protocol = selection_fixture(accuracy)
            self.assertEqual(report.verify_selection(selection, arms, protocol), expected)
        selection, arms, protocol = selection_fixture(.75)
        for field, value in (("score", .9), ("report_sha256", "d" * 64)):
            changed = copy.deepcopy(selection)
            changed["arms"]["sinusoidal"][field] = value
            with self.assertRaises(ValueError):
                report.verify_selection(changed, arms, protocol)
        changed = copy.deepcopy(selection)
        changed["arms"]["alibi"]["checkpoint"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            report.verify_selection(changed, arms, protocol)
        selection["selected_position"] = "alibi"
        with self.assertRaisesRegex(ValueError, "winner"):
            report.verify_selection(selection, arms, protocol)

    def test_three_arm_csv_full_and_zoom_share_exact_rows(self):
        summary = {"arms": {arm: {"curves": {str(step): {role: report.metric_rows(metric(step, role), arm)
                   for role in report.ROLES} for step in report.STEPS}} for arm in report.ARMS}}
        table = report.metric_table(summary)
        self.assertEqual(len(table), 432)
        for arm in report.ARMS:
            rows = report.endpoint_plot_rows(summary, arm)
            selected = [r for r in table if (r["arm"], r["update"], r["role"]) == (arm, 10000, "ood_dev")]
            self.assertEqual(rows, selected)
            self.assertEqual([r for r in rows if 10 <= r["length"] <= 18], selected[9:18])
            self.assertIs(rows, summary["arms"][arm]["curves"]["10000"]["ood_dev"])
        summary["arms"][report.ARMS[2]]["curves"]["10000"]["ood_dev"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            report.metric_table(summary)

    def test_history_and_frozen_source_guard_reject_missing_or_changed_inputs(self):
        history = [{"update": i, "examples_seen": i*1024,
                    "order_chain": hashlib.sha256(str(i).encode()).hexdigest(),
                    "seconds": .01, "state_loss": .8, "latent_loss": .2,
                    "weighted_latent_loss": .2, "loss": 1.0,
                    "token_accuracy": .5, "whole_word_exact": .25, "grad_norm": 1.0}
                   for i in range(1, 10001)]
        packet = {"order_chain": history[-1]["order_chain"], "train_seconds": 100.0, "elapsed_seconds": 101.0}
        report.validate_history(history, packet)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            report.validate_history(history[1:], packet)
        baseline = {f"source/{i}.py": "a" * 64 for i in range(46)}
        previous = {**baseline, **{key: "b" * 64 for key in report.PRIOR_VARIANT_SOURCES}}
        new = {**previous, **{key: "c" * 64 for key in report.NEW_SOURCES}}
        frozen = {46: report._digest_dict(baseline), 49: report._digest_dict(previous), 52: report._digest_dict(new)}
        with patch.dict(report.FROZEN_SOURCES, frozen):
            report.validate_source_lineage(baseline, new)
            changed = dict(new)
            changed[next(iter(baseline))] = "d" * 64
            with self.assertRaisesRegex(ValueError, "shared"):
                report.validate_source_lineage(baseline, changed)
            changed = dict(new)
            changed[next(iter(report.PRIOR_VARIANT_SOURCES))] = "d" * 64
            with self.assertRaisesRegex(ValueError, "49/52"):
                report.validate_source_lineage(baseline, changed)

    def test_evaluation_scalar_and_count_mismatches_rejected(self):
        value = metric(10000, "ood_dev")
        value["token_accuracy"] = .6
        with self.assertRaisesRegex(ValueError, "disagrees"):
            report.metric_rows(value, report.ARMS[2])
        value = metric(10000, "ood_dev")
        value["isolated_state_accuracy"][3] += 1e-6
        with self.assertRaisesRegex(ValueError, "integer"):
            report.metric_rows(value, report.ARMS[2])


if __name__ == "__main__":
    unittest.main()
