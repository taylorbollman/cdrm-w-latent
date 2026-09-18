"""Window resume, matched-budget restrictions, and exact plot/count semantics."""
import copy
import json

import pytest

from scripts import rt_a5_window_budget_report as report


def history(first,last):
    return [{"update":i,"order_chain":f"{i:064x}"} for i in range(first+1,last+1)]


def join_fixture():
    contract = {"schema":"rt-a5-window-training-v1","experiment_config":{"second_layer_window":2}}
    initial = {"baseline_initialization":{"canonical_sha256":"a"*64},"changed_parameter_slices":[]}
    sources = {"window.py":"b"*64}
    parent = {"path":str(report.ROOT / ".runtime/fixture/window-parent.pt"),"sha256":"c"*64}
    left = {"start_update":0,"completed_updates":10000,"contract":contract,"initialization":initial,"source_files":sources}
    right = {"start_update":10000,"endpoint":100000,"contract":contract,"initialization":initial,
             "source_files":sources,"parent_checkpoint":parent}
    pilot = {"report":left,"history":history(0,10000),"checkpoints":{"10000":parent},
             "input_files":{"report":{"sha256":"d"*64}}}
    continuation = {"report":right,"history":history(10000,100000)}
    protocol = {"strict_contract":contract,"source_files":sources,"parent_checkpoint_sha256":"c"*64,
                "parent_report_sha256":"d"*64}
    return pilot,continuation,protocol


def test_window_resume_joins_exact100k_without_reset_or_duplicate_tail():
    pilot,continuation,protocol = join_fixture()
    rows = report.verify_join(pilot,continuation,protocol,100000)
    assert len(rows) == 100000 and rows[-1]["update"] == 100000
    assert len(report.verify_join(pilot,continuation,protocol,20000)) == 20000
    for mutate in (
        lambda c:c["report"].update(start_update=0),
        lambda c:c["report"]["contract"]["experiment_config"].update(second_layer_window=None),
        lambda c:c["report"]["initialization"].update(changed_parameter_slices=["identity"]),
        lambda c:c["report"]["parent_checkpoint"].update(sha256="wrong"),
        lambda c:c["history"].insert(0,{"update":10000,"order_chain":"0"*64}),
    ):
        changed = copy.deepcopy(continuation)
        mutate(changed)
        with pytest.raises(ValueError):
            report.verify_join(pilot,changed,protocol,100000)


@pytest.mark.parametrize("status,error",[("running",None),("failed","KeyboardInterrupt"),("failed","RuntimeError")])
def test_final_window_report_never_accepts_stopped_or_incomplete100k(tmp_path,status,error):
    packet = {"schema":"rt-a5-window-training-v1","status":status,"error_type":error,
              "completed_updates":80000,"endpoint":100000,"start_update":10000}
    (tmp_path / "report.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError,match="must complete"):
        report.read_continuation(tmp_path,{},through_update=100000)


def test_window_cli_cannot_change_embedding_or_window_and_calls_original_recipe_guard(monkeypatch):
    called = []
    monkeypatch.setattr(report.budget,"verify_run_config",lambda config,protocol,pilot:called.append(pilot))
    report.validate_config({"position_encoding":"alibi","second_layer_window":2},{},pilot=False)
    assert called == [False]
    for config in ({"position_encoding":"sinusoidal","second_layer_window":2},
                   {"position_encoding":"alibi","second_layer_window":None}):
        with pytest.raises(ValueError,match="positions or direct-read"):
            report.validate_config(config,{},pilot=False)


def metric(step,role):
    length = 12 if role == "dev" else 36
    exact = ([.5]*12+[.25]*2+[0.]*22)[:length]
    return {"update":step,"role":role,"length":length,"rows":102400,"tokens":102400*length,
            "isolated_state_accuracy":[.5]*length,"cumulative_prefix_exactness":exact,
            "per_position_ce":[1.]*length,"ce":1.,"token_accuracy":.5,
            "whole_word_exact_match":exact[-1],"final_state_accuracy":.5}


def summary_fixture():
    result = {"matched_checkpoint_updates":[*report.STEPS,*(s for s in report.budget.NEW_STEPS if s <= 80000)],
              "arms":{},"budget":{"total_word_presentations":102400000,"nominal_training_passes":128},
              "training_seconds":1000.}
    for arm,endpoint in ((report.WINDOW,100000),(report.FULL,80000)):
        steps = [*report.STEPS,*(s for s in report.budget.NEW_STEPS if s <= endpoint)]
        metrics = {str(s):{role:metric(s,role) for role in report.ROLES} for s in steps}
        result["arms"][arm] = {"checkpoint_updates":steps,"metrics":metrics,
            "curves":{str(s):{role:report.metric_rows(metrics[str(s)][role],arm) for role in report.ROLES} for s in steps},
            "training_curve":[{"update":s,"state_ce":.8,"latent_loss":.2} for s in range(100,endpoint+1,100)]}
    result["checkpoint_summary"] = report.checkpoint_summary(result)
    return result


def test_full_and_zoom_use_identical_window_endpoint_rows_and_matched80k(tmp_path):
    summary = summary_fixture()
    rows = report.metric_table(summary)
    assert len(rows) == 1152
    for arm,step in ((report.WINDOW,10000),(report.WINDOW,100000),(report.FULL,80000),(report.WINDOW,80000)):
        selected = [r for r in rows if (r["arm"],r["update"],r["role"]) == (arm,step,"ood_dev")]
        plotted = report.plot_rows(summary,arm,step)
        assert plotted is summary["arms"][arm]["curves"][str(step)]["ood_dev"]
        assert selected == plotted and selected[9:18] == plotted[9:18]
        assert plotted[-1]["E"] == 0 and plotted[-1]["M"] == plotted[-1]["A"] == .5
    with pytest.raises(ValueError,match="unavailable"):
        report.plot_rows(summary,report.FULL,100000)
    points = {(r["arm"],r["update"]):r for r in summary["checkpoint_summary"]}
    assert points[(report.WINDOW,100000)]["matched_budget_available"] is False
    assert points[(report.WINDOW,80000)]["matched_budget_available"] is True
    text = report.markdown_report(summary)
    assert "no 100k comparator" in text and "after reviewing development curves" in text
    assert "unevaluated" in text and "prospectively fixed" in text
    figures = report.plot_results(summary,tmp_path)
    assert len(figures) == 5
    for files in figures.values():
        assert (tmp_path / files["png"]).read_bytes().startswith(b'\x89PNG')
        assert (tmp_path / files["pdf"]).read_bytes().startswith(b'%PDF')


def test_missing_prefix_or_unavailable_full100k_cannot_enter_export():
    summary = summary_fixture()
    summary["arms"][report.FULL]["curves"]["80000"]["ood_dev"][0]["update"] = 100000
    with pytest.raises(ValueError,match="leaked"):
        report.metric_table(summary)
    summary = summary_fixture()
    summary["arms"][report.WINDOW]["curves"]["100000"]["ood_dev"].pop()
    with pytest.raises(ValueError,match="Missing"):
        report.metric_table(summary)
    with pytest.raises(ValueError,match="every prefix"):
        report.plot_rows(summary,report.WINDOW,100000)
