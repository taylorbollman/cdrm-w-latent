"""Focused CPU-only guards for the combined pilot's comparison and plot data."""
import copy
import hashlib
import math
import unittest

from scripts.rt_a5_nextlat_variant_report import (
    ARMS, NEW_CONTRACT_FIELDS, ROLES, STEPS, TRAIN_SCHEMA,
    compare_contracts, endpoint_plot_rows, metric_rows, metric_table,
    validate_history,
)


def contracts():
    baseline = {"schema": "rt-a5-nextlat-training-v1", "source_sha256": "0" * 64,
                "objective": {"latent_weight": 1.0}, "optimizer": "unchanged-adam",
                "data_manifest_sha256": "1" * 64, "seed": 1234,
                "model_config": {"alibi": True, "rope": False, "n_layers": 2}}
    variant = copy.deepcopy(baseline)
    variant.update(schema=TRAIN_SCHEMA, source_sha256="2" * 64, **NEW_CONTRACT_FIELDS)
    variant["model_config"]["alibi"] = False
    variant["variant_config"] = {
        "schema": "rt-a5-nextlat-variant-config-v1", "name": "value-identity-sinusoidal",
        "architecture": "rt", "width": 512,
        "initialization": {"kind": "value_path_identity", "noise_multiplier": 1.0,
                           "noise_seed": 1236, "noise_std": 1/math.sqrt(512),
                           "matrices": ["value_projection", "attention_output_projection"],
                           "layers": "both recurrent blocks"},
        "position_encoding": {"kind": "fixed_sinusoidal", "base": 10000.0, "amplitude": 1.0,
                              "position_origin": 0, "token_embedding_scale": 1.0,
                              "alibi": False, "rope": False, "learned_position_parameters": False,
                              "persistent_position_buffers": False,
                              "nextlat_conditioning": "raw next-operation token embedding without positional addition"}}
    return baseline, variant


def metric(step, role):
    n, length = 102400, {"dev": 12, "ood_dev": 36}[role]
    return {"update": step, "role": role, "rows": n, "length": length, "tokens": n*length,
            "isolated_state_accuracy": [.5]*length, "cumulative_prefix_exactness": [.5]*length,
            "per_position_ce": [1.0]*length, "token_accuracy": .5, "whole_word_exact_match": .5,
            "final_state_accuracy": .5, "ce": 1.0}


def summary_fixture():
    return {"arms": {arm: {"curves": {str(step): {role: metric_rows(metric(step, role), arm)
              for role in ROLES} for step in STEPS}} for arm in ARMS}}


class VariantReportChecks(unittest.TestCase):
    def test_only_planned_contract_changes_allowed(self):
        baseline, variant = contracts()
        compare_contracts(baseline, variant)
        for field, value in (("optimizer", "different"), ("seed", 7),
                             ("data_manifest_sha256", "3" * 64), ("objective", {"latent_weight": 2})):
            changed = copy.deepcopy(variant)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "Unplanned"):
                compare_contracts(baseline, changed)

    def test_learned_position_and_wrong_noise_rejected(self):
        baseline, variant = contracts()
        changed = copy.deepcopy(variant)
        changed["variant_config"]["position_encoding"]["learned_position_parameters"] = True
        with self.assertRaises(ValueError):
            compare_contracts(baseline, changed)
        variant["variant_config"]["initialization"]["noise_std"] /= 100
        with self.assertRaisesRegex(ValueError, "variance"):
            compare_contracts(baseline, variant)

    def test_csv_full_and_zoom_share_exact_rows(self):
        summary = summary_fixture()
        table = metric_table(summary)
        self.assertEqual(len(table), 288)
        for arm in ARMS:
            full = endpoint_plot_rows(summary, arm)
            selected = [r for r in table if (r["arm"], r["update"], r["role"]) == (arm, 10000, "ood_dev")]
            self.assertEqual(full, selected)
            self.assertEqual([r for r in full if 10 <= r["length"] <= 18], selected[9:18])
            self.assertIs(full, summary["arms"][arm]["curves"]["10000"]["ood_dev"])
        summary["arms"][ARMS[1]]["curves"]["10000"]["ood_dev"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            metric_table(summary)

    def test_history_rejects_missing_update_or_wrong_objective(self):
        history = [{"update": i, "examples_seen": i*1024,
                    "order_chain": hashlib.sha256(str(i).encode()).hexdigest(),
                    "seconds": .01, "state_loss": .8, "latent_loss": .2,
                    "weighted_latent_loss": .2, "loss": 1.0,
                    "token_accuracy": .5, "whole_word_exact": .25, "grad_norm": 1.0}
                   for i in range(1, 10001)]
        report = {"order_chain": history[-1]["order_chain"], "train_seconds": 100.0, "elapsed_seconds": 101.0}
        validate_history(history, report)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_history(history[1:], report)
        history[77]["weighted_latent_loss"] = .3
        with self.assertRaisesRegex(ValueError, "weight-one"):
            validate_history(history, report)

    def test_evaluation_scalar_and_count_mismatches_rejected(self):
        value = metric(10000, "ood_dev")
        value["token_accuracy"] = .6
        with self.assertRaisesRegex(ValueError, "disagrees"):
            metric_rows(value, ARMS[1])
        value = metric(10000, "ood_dev")
        value["isolated_state_accuracy"][3] += 1e-6
        with self.assertRaisesRegex(ValueError, "integer"):
            metric_rows(value, ARMS[1])


if __name__ == "__main__":
    unittest.main()
