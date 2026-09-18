"""Explicit-stop interpretation, retained-prefix accounting, and shared graph rows."""
import copy

import pytest

from scripts import rt_a5_nextlat_stopped_report as report


def stopped_fixture():
    stop = 80002
    history = [{"update":i, "examples_seen":i*1024, "order_chain":f"{i:064x}",
                "seconds":.01, "loss":1., "state_loss":.8, "latent_loss":.2,
                "weighted_latent_loss":.2, "token_accuracy":.5, "whole_word_exact":.25,
                "grad_norm":.5} for i in range(10001, stop+1)]
    seconds = sum(r["seconds"] for r in history[:70000])
    raw = {"schema":report.budget.TRAIN_SCHEMA, "status":"failed", "error_type":"KeyboardInterrupt",
           "wandb":{"status":"synced_failed_experiment"}, "start_update":10000,
           "completed_updates":stop, "endpoint":100000, "confirmation_evaluated":False,
           "latent_rollout_evaluated":False, "order_chain":history[69999]["order_chain"],
           "train_seconds":seconds, "elapsed_seconds":900.}
    revision = {"raw_training_report":{"status":"failed", "error_type":"KeyboardInterrupt",
                 "wandb_status":"synced_failed_experiment", "completed_updates":stop, "configured_endpoint":100000},
                "raw_history":{"first_update":10001, "last_update":stop, "updates":70002},
                "accepted_continuation":{"first_update":10001, "last_update":80000, "updates":70000,
                                         "order_chain":raw["order_chain"], "training_seconds":seconds},
                "excluded_tail":{"first_update":80001, "last_completed_update":stop, "updates":2, "training_seconds":.02}}
    return history, raw, revision


def test_explicit_interrupt_keeps_accepted_checkpoint_counters_and_unsaved_tail_separate():
    history, raw, revision = stopped_fixture()
    accepted, tail = report.validate_stopped_history(history, raw, revision)
    assert len(accepted) == 70000 and len(tail) == 2
    assert accepted[-1]["update"] == 80000 and tail[-1]["update"] == 80002
    assert raw["completed_updates"] == 80002
    assert raw["order_chain"] != history[-1]["order_chain"]
    assert sum(r["seconds"] for r in tail) == .02


@pytest.mark.parametrize("mutation", [
    lambda raw, revision: raw.update(error_type="RuntimeError"),
    lambda raw, revision: raw.update(status="complete"),
    lambda raw, revision: raw["wandb"].update(status="synced"),
    lambda raw, revision: revision["raw_training_report"].update(completed_updates=80003),
    lambda raw, revision: revision["excluded_tail"].update(updates=3),
    lambda raw, revision: revision["accepted_continuation"].update(training_seconds=1),
    lambda raw, revision: raw.update(order_chain=f"{80002:064x}"),
])
def test_arbitrary_failure_relabeling_or_counter_tampering_rejected(mutation):
    history, raw, revision = stopped_fixture()
    mutation(raw, revision)
    with pytest.raises(ValueError):
        report.validate_stopped_history(history, raw, revision)


def test_excluded_tail_is_preserved_but_still_requires_finite_correct_history():
    history, raw, revision = stopped_fixture()
    history[-1]["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        report.validate_stopped_history(history, raw, revision)
    history[-1]["loss"] = 1.
    history[-1]["update"] -= 1
    with pytest.raises(ValueError, match="exactly once"):
        report.validate_stopped_history(history, raw, revision)


def test_stop_revision_requires_explicit_user_scope_and_exact_file_hashes(tmp_path):
    revision = {"schema":"rt-a5-nextlat-endpoint-revision-v1", "action":"stop_at_retained_checkpoint",
                "original_endpoint":100000, "accepted_endpoint":80000,
                "selection_kind":"user_selected_after_development_review", "confirmation_evaluated":False,
                "latent_rollout_evaluated":False, "expected_new_checkpoints":list(report.NEW_STEPS),
                "user_instruction":"Please stop at the80k checkpoint.",
                "termination":{"signal":"SIGINT", "exit_code":1, "trainer_process_gone":True, "container_gone":True}}
    directory = tmp_path / "train"
    (directory / "checkpoints").mkdir(parents=True)
    for key, path in (("protocol",tmp_path / "protocol.json"), ("raw_training_report",directory / "report.json"),
                      ("raw_history",directory / "history.jsonl"), ("accepted_checkpoint",directory / "checkpoints/step-080000.pt")):
        path.write_bytes(b'fixture\n')
        revision[key] = report.hash_file(path)
    revision["accepted_checkpoint"]["completed_updates"] = 80000
    exit_path = tmp_path / "exit.txt"
    exit_path.write_text('1\n')
    revision["termination"]["exit_record"] = report.hash_file(exit_path)
    report.validate_revision(revision, tmp_path / "protocol.json", directory)
    for key,value in (("selection_kind","prospective_fixed"), ("accepted_endpoint",100000), ("user_instruction","")):
        changed = copy.deepcopy(revision)
        changed[key] = value
        with pytest.raises(ValueError):
            report.validate_revision(changed, tmp_path / "protocol.json", directory)
    (directory / "report.json").write_bytes(b'changed failure evidence')
    with pytest.raises(ValueError, match="hash/size"):
        report.validate_revision(revision, tmp_path / "protocol.json", directory)


def metric(step, role):
    n, length = 102400, 12 if role == "dev" else 36
    exact = ([.5]*12 + [.25]*2 + [0.]*22)[:length]
    return {"update":step, "role":role, "rows":n, "length":length, "tokens":n*length,
            "isolated_state_accuracy":[.5]*length, "cumulative_prefix_exactness":exact,
            "per_position_ce":[1.]*length, "ce":1., "token_accuracy":.5,
            "whole_word_exact_match":exact[-1], "final_state_accuracy":.5}


def test_plots_csv_and_labels_use_80k_and_never_leak_raw_tail_or_pure_rt_100k(tmp_path):
    steps = [*report.STEPS,*report.NEW_STEPS]
    metrics = {str(s):{role:metric(s,role) for role in report.ROLES} for s in steps}
    curves = {str(s):{role:report.metric_rows(metrics[str(s)][role],"rt_nextlat") for role in report.ROLES} for s in steps}
    pure = {"metrics":{str(s):metrics[str(s)] for s in (10000,80000)},
            "curves":{str(s):{role:report.metric_rows(metrics[str(s)][role],"rt") for role in report.ROLES} for s in (10000,80000)}}
    summary = {"checkpoint_updates":steps, "reported_through_update":80000, "curves":curves,
               "historical_metrics":metrics, "pure_rt_reference":pure,
               "training_curve":[{"update":s,"state_ce":.8,"latent_loss":.2} for s in range(100,80001,100)],
               "stop":{"raw_completed_updates":81607,"excluded_completed_updates":1607,
                       "excluded_evaluation_records":6,"excluded_one_step_diagnostic_records":6},
               "timing":{"accepted_total_training_seconds":800.,"accepted_continuation_training_seconds":700.,
                         "excluded_tail_training_seconds":16.,"raw_continuation_training_seconds":716.,
                         "raw_continuation_job_elapsed_seconds":740.}}
    summary["checkpoint_summary"] = report.checkpoint_summary(summary)
    rows = report.metric_table(summary)
    assert len(rows) == 624 and max(r["update"] for r in rows) == 80000
    plotted = report.budget.endpoint_plot_rows(summary,80000)
    selected = [r for r in rows if (r["arm"],r["update"],r["role"]) == ("rt_nextlat",80000,"ood_dev")]
    assert plotted is curves["80000"]["ood_dev"]
    assert selected == plotted and selected[9:18] == plotted[9:18]
    assert plotted[-1]["E"] == 0 and plotted[-1]["A"] == plotted[-1]["M"] == .5
    text = report.markdown_report(summary)
    assert "after reviewing development curves" in text
    assert "retrospective user-selected endpoint" in text
    assert "Pure RT" in text and "not substituted for 80k" in text
    assert "1,607" in text and "unevaluated" in text
    figures = report.budget.plot_results(summary,tmp_path)
    assert all((tmp_path / value["png"]).read_bytes().startswith(b'\x89PNG') for value in figures.values())
    assert all((tmp_path / value["pdf"]).read_bytes().startswith(b'%PDF') for value in figures.values())
    summary["pure_rt_reference"]["curves"]["80000"]["ood_dev"][0]["update"] = 100000
    with pytest.raises(ValueError,match="leaked"):
        report.metric_table(summary)
