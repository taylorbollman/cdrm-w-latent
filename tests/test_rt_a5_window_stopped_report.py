"""Window-specific explicit-stop validation and retrospective80k plot consistency."""
import copy

import pytest

from scripts import rt_a5_window_stopped_report as report


def stopped_fixture():
    stop = 80002
    history = [{"update":i,"examples_seen":i*1024,"order_chain":f"{i:064x}","seconds":.01,
                "loss":1.,"state_loss":.8,"latent_loss":.2,"weighted_latent_loss":.2,
                "token_accuracy":.5,"whole_word_exact":.25,"grad_norm":.5} for i in range(10001,stop+1)]
    seconds = sum(r["seconds"] for r in history[:70000])
    packet = {"schema":"rt-a5-window-training-v1","status":"failed","error_type":"KeyboardInterrupt",
        "wandb":{"status":"synced_failed_experiment"},"start_update":10000,"completed_updates":stop,
        "endpoint":100000,"confirmation_evaluated":False,"latent_rollout_evaluated":False,
        "order_chain":history[69999]["order_chain"],"train_seconds":seconds,"elapsed_seconds":900.}
    revision = {"schema":report.REVISION_SCHEMA,
        "raw_training_report":{"status":"failed","error_type":"KeyboardInterrupt","wandb_status":"synced_failed_experiment",
                               "completed_updates":stop,"configured_endpoint":100000},
        "raw_history":{"first_update":10001,"last_update":stop,"updates":70002},
        "accepted_continuation":{"first_update":10001,"last_update":80000,"updates":70000,"training_seconds":seconds,
                                 "order_chain":packet["order_chain"]},
        "excluded_tail":{"first_update":80001,"last_completed_update":stop,"updates":2,"training_seconds":.02}}
    return history,packet,revision


def test_window_stop_adapts_common_history_guard_without_mutating_raw_evidence():
    history,packet,revision = stopped_fixture()
    raw_copy = copy.deepcopy(packet)
    accepted,tail = report.validate_stopped_history(history,packet,revision)
    assert len(accepted) == 70000 and len(tail) == 2
    assert accepted[-1]["update"] == 80000 and tail[-1]["update"] == 80002
    assert packet == raw_copy and packet["schema"] == "rt-a5-window-training-v1"
    assert packet["order_chain"] != history[-1]["order_chain"]


@pytest.mark.parametrize("field,value",[("schema","rt-a5-nextlat-training-v1"),("error_type","RuntimeError"),
                                       ("status","complete"),("completed_updates",80003)])
def test_nonwindow_schema_arbitrary_failure_or_counter_change_rejected(field,value):
    history,packet,revision = stopped_fixture()
    packet[field] = value
    with pytest.raises(ValueError):
        report.validate_stopped_history(history,packet,revision)


def test_window_revision_schema_is_required_before_common_stop_guards(monkeypatch,tmp_path):
    seen = []
    monkeypatch.setattr(report.stopped,"validate_revision",lambda revision,protocol,directory:seen.append(revision))
    value = {"schema":report.REVISION_SCHEMA,"action":"stop_at_retained_checkpoint"}
    report.validate_revision(value,tmp_path / "protocol.json",tmp_path)
    assert seen[0]["schema"] == "rt-a5-nextlat-endpoint-revision-v1"
    assert value["schema"] == report.REVISION_SCHEMA
    for schema in (None,"rt-a5-nextlat-endpoint-revision-v1","rt-a5-window-budget-protocol-v1"):
        with pytest.raises(ValueError,match="window user-stop"):
            report.validate_revision({**value,"schema":schema},tmp_path / "protocol.json",tmp_path)


def metric(step,role):
    length = 12 if role == "dev" else 36
    exact = ([.5]*12+[.25]*2+[0.]*22)[:length]
    return {"update":step,"role":role,"rows":102400,"length":length,"tokens":102400*length,
        "isolated_state_accuracy":[.5]*length,"cumulative_prefix_exactness":exact,
        "per_position_ce":[1.]*length,"ce":1.,"token_accuracy":.5,"whole_word_exact_match":exact[-1],"final_state_accuracy":.5}


def summary_fixture():
    steps = [*report.STEPS,*report.NEW_STEPS]
    summary = {"matched_checkpoint_updates":steps,"arms":{},
        "stop":{"raw_completed_updates":85376,"excluded_completed_updates":5376,
                "excluded_evaluation_records":20,"excluded_one_step_diagnostic_records":20},
        "timing":{"accepted_total_training_seconds":800.,"accepted_continuation_training_seconds":700.,
            "excluded_tail_training_seconds":50.,"raw_continuation_training_seconds":750.,"raw_continuation_job_elapsed_seconds":900.}}
    for arm in (report.WINDOW,report.FULL):
        metrics = {str(s):{role:metric(s,role) for role in report.ROLES} for s in steps}
        summary["arms"][arm] = {"checkpoint_updates":steps,"metrics":metrics,
            "curves":{str(s):{role:report.metric_rows(metrics[str(s)][role],arm) for role in report.ROLES} for s in steps},
            "training_curve":[{"update":s,"state_ce":.8,"latent_loss":.2} for s in range(100,80001,100)]}
    summary["checkpoint_summary"] = report.continuing.checkpoint_summary(summary)
    return summary


def test_window10k80k_and_matched80k_graphs_share_exact_rows_and_qualification(tmp_path):
    summary = summary_fixture()
    rows = report.metric_table(summary)
    assert len(rows) == 1056 and max(r["update"] for r in rows) == 80000
    for arm,step in ((report.WINDOW,10000),(report.WINDOW,80000),(report.FULL,80000)):
        plotted = report.plot_rows(summary,arm,step)
        selected = [r for r in rows if (r["arm"],r["update"],r["role"]) == (arm,step,"ood_dev")]
        assert plotted is summary["arms"][arm]["curves"][str(step)]["ood_dev"]
        assert plotted == selected and plotted[9:18] == selected[9:18]
        assert plotted[-1]["E"] == 0 and plotted[-1]["A"] == plotted[-1]["M"] == .5
    text = report.markdown_report(summary)
    assert "retrospective" in text and "after reviewing development curves" in text
    assert "neither model has a completed 100k outcome" in text
    assert "5,376" in text and "unevaluated" in text
    figures = report.plot_results(summary,tmp_path)
    assert len(figures) == 5
    for paths in figures.values():
        assert (tmp_path / paths["png"]).read_bytes().startswith(b'\x89PNG')
        assert (tmp_path / paths["pdf"]).read_bytes().startswith(b'%PDF')


def test_no_arm_can_plot_or_export_tail_or_unavailable100k():
    summary = summary_fixture()
    for arm in (report.WINDOW,report.FULL):
        with pytest.raises(ValueError,match="no 100k"):
            report.plot_rows(summary,arm,100000)
    summary["arms"][report.WINDOW]["curves"]["80000"]["ood_dev"][0]["update"] = 85376
    with pytest.raises(ValueError,match="leaked"):
        report.metric_table(summary)
