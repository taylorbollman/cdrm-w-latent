import json
from types import SimpleNamespace

import pytest

from scripts import olmo_campaign_rank_failure as failure


def args_for(tmp_path):
    root = tmp_path / "checkpoints"
    checkpoint = root / "update-000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text('{"completed": 1}\n')
    receipt = {"directory": str(checkpoint), "counters": {"optimizer_updates": 1},
               "manifest_sha256": failure.sha256_file(checkpoint / "manifest.json"),
               "retention": {"download_sha256_verified": True}}
    (tmp_path / "latest-checkpoint.json").write_text(json.dumps(receipt))
    return SimpleNamespace(output_dir=tmp_path, checkpoint_root=root, resume=None,
                           inject_log_error_at=None, request_stop_after=None, storage_prefix="gs://test")


def test_rank1_exits_before_second_backward_and_keeps_committed_authority(tmp_path):
    args = args_for(tmp_path)
    events = []
    class Runner:
        def backward(self, value):
            events.append(value)
            return value
    original = Runner.backward
    def stop(code):
        assert code == 73
        raise SystemExit(code)
    with failure.inject_exit(args, SimpleNamespace(rank=1), enabled=True, runner_class=Runner, exit_process=stop):
        runner = Runner()
        assert runner.backward("update1") == "update1"
        with pytest.raises(SystemExit) as caught:
            runner.backward("update2")
        assert caught.value.code == 73
    assert events == ["update1"] and Runner.backward is original
    marker = json.loads((tmp_path / "rank-1-deliberate-exit.json").read_text())
    assert marker["last_committed_update"] == 1 and marker["backward_invocation"] == 2
    assert not (args.checkpoint_root / "update-000002").exists()


@pytest.mark.parametrize("rank,enabled", [(0, True), (1, False)])
def test_reference_and_peer_paths_leave_backward_unchanged(tmp_path, rank, enabled):
    args = args_for(tmp_path)
    class Runner:
        def backward(self, value):
            return value * 2
    with failure.inject_exit(args, SimpleNamespace(rank=rank), enabled=enabled, runner_class=Runner,
                             exit_process=lambda _: pytest.fail("unexpected exit")):
        runner = Runner()
        assert [runner.backward(i) for i in range(3)] == [0, 2, 4]
    assert not (tmp_path / "rank-1-deliberate-exit.json").exists()


def test_unretained_boundary_rejects_without_exit_marker(tmp_path):
    args = args_for(tmp_path)
    path = tmp_path / "latest-checkpoint.json"
    receipt = json.loads(path.read_text()); receipt["retention"] = {}
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="authenticated"):
        failure.write_exit_marker(args, rank=1, call=2)
    assert not (tmp_path / "rank-1-deliberate-exit.json").exists()


def test_sources_extend_frozen_guarded_inventory():
    sources = failure.source_hashes()
    assert all(sources[name] == digest for name, digest in failure.guarded.source_hashes().items())
