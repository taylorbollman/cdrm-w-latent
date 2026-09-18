"""Depth, window location, initialization and prospective versus stopped evidence guards."""
import copy
import hashlib
import json

import pytest

from scripts import rt_a5_depth_order_report as report


def metric(step, role, rows=102400):
    length = 12 if role == "dev" else 36
    exact = ([.5]*12+[.25]*2+[0.]*22)[:length]
    return {"update":step,"role":role,"rows":rows,"length":length,"tokens":rows*length,
        "route":"backbone_only","isolated_state_accuracy":[.5]*length,"cumulative_prefix_exactness":exact,
        "per_position_ce":[1.]*length,"ce":1.,"token_accuracy":.5,"whole_word_exact_match":exact[-1],"final_state_accuracy":.5}


def summary_fixture():
    summary = {"arms":{},"reference_provenance":{
        "full":{"stop":{"raw_completed_updates":81607,"excluded_completed_updates":1607}},
        "window_second":{"stop":{"raw_completed_updates":85376,"excluded_completed_updates":5376}}}}
    for arm in report.ARMS:
        metrics = {str(s):{role:metric(s,role) for role in report.ROLES} for s in report.STEPS}
        summary["arms"][arm] = {"checkpoint_updates":list(report.STEPS),"metrics":metrics,
            "parameter_count":13702656 if arm == "seq4_alibi" else 7407104,
            "selection_kind":"prospectively_fixed80k" if arm in report.FRESH else "user_selected_after_development_review",
            "curves":{str(s):{role:report.metric_rows(metrics[str(s)][role],arm) for role in report.ROLES} for s in report.STEPS},
            "training_curve":[{"update":s,"state_ce":.8,"latent_loss":.2} for s in range(100,80001,100)]}
    summary["checkpoint_summary"] = report.checkpoint_summary(summary)
    return summary


def test_all_four_matched80k_curves_export_identical_full_and_boundary_rows(tmp_path):
    summary = summary_fixture()
    rows = report.metric_table(summary)
    assert len(rows) == 2112 and len(summary["checkpoint_summary"]) == 44
    for arm in report.ARMS:
        for step in (10000,80000):
            plotted = report.plot_rows(summary,arm,step)
            selected = [r for r in rows if (r["arm"],r["update"],r["role"]) == (arm,step,"ood_dev")]
            assert plotted is summary["arms"][arm]["curves"][str(step)]["ood_dev"]
            assert plotted == selected and plotted[9:18] == selected[9:18]
            assert plotted[-1]["E"] == 0 and plotted[-1]["A"] == plotted[-1]["M"] == .5
    text = report.markdown_report(summary)
    for phrase in ("13,702,656", "7,407,104", "not a parameter- or FLOP-matched", "canonical four-layer",
                   "retrospective", "fixed before training", "1,607", "5,376", "completed 100k outcome", "unevaluated"):
        assert phrase in text
    figures = report.plot_results(summary,tmp_path)
    assert set(figures) == {"length-full","length-boundary","length-boundary-10k","exactness-vs-updates","training-losses"}
    for files in figures.values():
        assert (tmp_path / files["png"]).read_bytes().startswith(b'\x89PNG')
        assert (tmp_path / files["pdf"]).read_bytes().startswith(b'%PDF')


def test_unsaved_reference_tail_and_unavailable100k_cannot_enter_metric_exports():
    summary = summary_fixture()
    for arm in report.ARMS:
        with pytest.raises(ValueError,match="retained matched"):
            report.plot_rows(summary,arm,100000)
    summary["arms"][report.REFERENCES[1]]["curves"]["80000"]["ood_dev"][0]["update"] = 85376
    with pytest.raises(ValueError,match="leaked"):
        report.metric_table(summary)


def pairing_fixture(variant):
    base = {"canonical_sha256":"original_rt2","predictor_sha256":"same_predictor","predictor_config":{"width":512}}
    contract = {"architecture":"rt","schema":"old","source_sha256":"old","model_config":{"n_layers":2,
        "block_type":"recurrent","d_model":512},"objective":{"target_detached":True,"latent_weight":1.},
        "optimizer":"original Adam","batch_size":1024}
    history = [{"order_chain":str(i)} for i in range(4)]
    reference = {"initialization":base,"contract":contract,"history":history}
    fresh = copy.deepcopy(reference)
    fresh["contract"].update(schema=report.TRAIN_SCHEMA,source_sha256=report.SOURCE_SHA,variant=variant)
    if variant == "rt_window2_first":
        fresh["initialization"].update(reference_two_layer_initialization=base,exact_two_layer_learned_initialization=True,
                                       changed_parameter_slices=[])
    else:
        fresh["contract"].update(architecture="seq")
        fresh["contract"]["model_config"].update(n_layers=4,block_type="sequential")
        fresh["initialization"].update(canonical_sha256="new_seq4",exact_two_layer_learned_initialization=False,
            changed_parameter_slices=None,backbone={"two_layer_initialization_identity_claimed":False})
    return fresh,reference


@pytest.mark.parametrize("variant",report.FRESH)
def test_only_declared_depth_and_architecture_differences_are_removed(variant):
    fresh,reference = pairing_fixture(variant)
    before = copy.deepcopy(fresh)
    report.compare_arm(variant,fresh,reference)
    assert fresh == before
    fresh["contract"]["objective"]["latent_weight"] = .5
    with pytest.raises(ValueError,match="objective or optimizer"):
        report.compare_arm(variant,fresh,reference)


@pytest.mark.parametrize("variant",report.FRESH)
def test_every_update_order_and_predictor_identity_must_match(variant):
    fresh,reference = pairing_fixture(variant)
    fresh["history"][1]["order_chain"] = "different"
    with pytest.raises(ValueError,match="minibatch order"):
        report.compare_arm(variant,fresh,reference)
    fresh,reference = pairing_fixture(variant)
    fresh["initialization"]["predictor_sha256"] = "different"
    with pytest.raises(ValueError,match="predictor initialization"):
        report.compare_arm(variant,fresh,reference)


@pytest.mark.parametrize("variant",report.FRESH)
def test_seq4_cannot_claim_rt2_initialization_and_rt2_cannot_change_learned_tensors(variant):
    fresh,reference = pairing_fixture(variant)
    if variant == "seq4_alibi":
        fresh["initialization"]["exact_two_layer_learned_initialization"] = True
    else:
        fresh["initialization"]["changed_parameter_slices"] = ["attention projection"]
    with pytest.raises(ValueError,match="initialization"):
        report.compare_arm(variant,fresh,reference)


def fresh_fixture(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{}')
    contract = {"data_manifest_sha256":hashlib.sha256(b'{}').hexdigest()}
    config = {"data_dir":str(data)}
    history = [{"update":i,"examples_seen":i*1024,"order_chain":f"{i:064x}","seconds":.01,
        "loss":1.,"state_loss":.8,"latent_loss":.2,"weighted_latent_loss":.2,
        "token_accuracy":.5,"whole_word_exact":.25,"grad_norm":.5} for i in range(1,1001)]
    checkpoints = []
    for step in (0,1000):
        path = directory / f"step-{step}.pt"
        path.write_bytes(f"opaque checkpoint {step}".encode())
        checkpoints.append({**report.hash_file(path),"completed_updates":step,"examples_seen":step*1024})
    packet = {"schema":report.TRAIN_SCHEMA,"status":"running","start_update":0,"parent_checkpoint":None,
        "endpoint":80000,"completed_updates":1000,"wandb":{"status":"running"},"contract":contract,
        "initialization":{"identity":"frozen"},"source_files":{},"confirmation_evaluated":False,
        "latent_rollout_evaluated":False,"order_chain":history[-1]["order_chain"],"train_seconds":sum(r["seconds"] for r in history),
        "elapsed_seconds":20.,"checkpoints":checkpoints,"evaluations":[],"one_step_diagnostics":[]}
    for step in (500,1000):
        for role in report.ROLES:
            packet["evaluations"].append(metric(step,role,102400 if step == 1000 else 4096))
            packet["one_step_diagnostics"].append({"update":step,"role":role,"rows":1024,
                "length":12 if role == "dev" else 36,"route":"teacher_conditioned_one_step_diagnostics",
                "loss":1.,"state_loss":.8,"latent_loss":.2,"weighted_latent_loss":.2,
                "diagnostics":{k:.1 for k in report.budget.DIAGNOSTICS}})
    protocol = {"source_files":{},"arms":{"seq4_alibi":{"directory":str(directory),"strict_contract":contract,
        "initialization":packet["initialization"],"resolved_args":config,"parameter_count":13702656,"parameter_tensors":39}}}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "report.json").write_text(json.dumps(packet))
    (directory / "history.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in history))
    return directory,protocol,packet


def test_live1k_snapshot_reads_only_saved_prefix_and_validates_checkpoint_bytes(tmp_path):
    directory,protocol,packet = fresh_fixture(tmp_path)
    with (directory / "history.jsonl").open('a') as stream:
        stream.write('{"update":1001,"loss":"unfinished live tail"}\n')
    arm = report.read_fresh("seq4_alibi",protocol,through_update=1000)
    assert len(arm["history"]) == 1000 and arm["checkpoint_updates"] == [1000]
    assert len(arm["training_curve"]) == 10
    (directory / "step-1000.pt").write_bytes(b'changed saved checkpoint')
    with pytest.raises(ValueError):
        report.read_fresh("seq4_alibi",protocol,through_update=1000)


@pytest.mark.parametrize("field,value",[("status","failed"),("start_update",10000),("parent_checkpoint",{"sha256":"parent"}),
    ("endpoint",100000),("completed_updates",79999),("completed_updates",80001),("wandb",{"status":"offline"})])
def test_final80k_requires_fresh_complete_synced_endpoint_without_failure_bypass(tmp_path,field,value):
    directory,protocol,packet = fresh_fixture(tmp_path)
    packet.update(status="complete",completed_updates=80000,wandb={"status":"synced"})
    packet[field] = value
    (directory / "report.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        report.read_fresh("seq4_alibi",protocol)


def test_nonfinite_history_and_changed_source_are_rejected_on_bounded_read(tmp_path):
    directory,protocol,packet = fresh_fixture(tmp_path)
    history = (directory / "history.jsonl").read_text().splitlines()
    row = json.loads(history[10]); row["latent_loss"] = float('nan'); history[10] = json.dumps(row)
    (directory / "history.jsonl").write_text('\n'.join(history)+'\n')
    with pytest.raises(ValueError,match="Nonfinite"):
        report.read_fresh("seq4_alibi",protocol,through_update=1000)
    packet["source_files"] = {"unexpected.py":"0"*64}
    (directory / "report.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError,match="frozen sources"):
        report.read_fresh("seq4_alibi",protocol,through_update=1000)


def test_standalone_package_preserves_pure_rt_order_reports_and_histories_without_checkpoint_bytes():
    record = lambda name:{"path":"/saved/"+name,"sha256":"0"*64,"bytes":1}
    summary = {"protocol_input":record("protocol.json"),"arms":{arm:{"inputs":{}} for arm in report.ARMS},
        "protocol":{name:record(name+".json") for name in ("data_manifest","preflight")},"reference_provenance":{}}
    for reference in ("full","window_second"):
        summary["reference_provenance"][reference] = {"revision_input":record("endpoint-revision.json"),
            "revision_evidence":{"protocol":record("protocol.json"),"termination_exit":record("training-exit-code.txt")}}
    summary["reference_provenance"]["full"]["pure_rt_order_inputs"] = {
        phase:{"report":record("report.json"),"history":record("history.jsonl"),"run_config":record("config.json"),
               "endpoint_checkpoint":record("checkpoint.pt")} for phase in ("pilot","continuation")}
    records = report.input_records(summary)
    retained = {str(destination) for destination,_ in records}
    for phase in ("pilot","continuation"):
        for name in ("report.json","history.jsonl","config.json"):
            assert f"inputs/pure-rt-order/{phase}/{name}" in retained
    assert not any(path.endswith('.pt') for path in retained)
