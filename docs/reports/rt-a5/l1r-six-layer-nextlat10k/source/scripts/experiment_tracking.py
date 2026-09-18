"""Optional online W&B tracking, separate from numerical experiment state.

Callers may supply a context manager that preserves their RNG state. No W&B
dependency is imported until tracking starts; credentials come from the process
environment and are never included in the configuration or error messages here.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path


def add_wandb_arguments(parser):
    parser.add_argument("--wandb-project", help="Enable required online W&B logging to this project")
    parser.add_argument("--wandb-entity", default="taylorbollman")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-run-name")


def scalar_metrics(values, prefix=""):
    """Flatten numeric leaves of a report without sending tensors or large arrays."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (bool, int, float)):
            result[name] = value
    return result


class OnlineTracker:
    """An explicitly online run whose SDK failures fail the calling experiment.

    `record` is JSON compatible and can be attached directly to a local report.
    `log` accepts metric dictionaries and `summary` accepts final report values,
    so the same helper serves numerical diagnostics and operational runs.
    """

    def __init__(self, *, project, output_dir, entity="taylorbollman", group=None,
                 name=None, preserve_state=nullcontext):
        self.output_dir = Path(output_dir)
        self.preserve_state = preserve_state
        self._run = None
        self.record = {"enabled": True, "mode": "online", "project": project,
                       "entity": entity, "group": group, "name": name,
                       "system_stats": False, "status": "pending"}

    def _call(self, operation, function):
        try:
            with self.preserve_state():
                return function()
        except Exception as error:
            self.record.update(status="failed", failed_operation=operation,
                               error_type=type(error).__name__)
            # SDK exception strings can contain network/authentication details.
            raise RuntimeError(f"Required online W&B {operation} failed "
                               f"({type(error).__name__}); local records are retained") from None

    def start(self, config):
        def initialize():
            import wandb
            if wandb.run is not None:
                raise RuntimeError("An existing W&B run would mix experiment lineages")
            self._run = wandb.init(
                project=self.record["project"], entity=self.record["entity"],
                group=self.record["group"], name=self.record["name"],
                mode="online", config=config, dir=str(self.output_dir),
                settings=wandb.Settings(init_timeout=60, save_code=False, disable_git=True,
                                        x_disable_stats=True))
            if self._run is None or self._run.settings.mode != "online" or not self._run.url:
                raise RuntimeError("W&B did not create an online run")
            self._run.define_metric("update")
            self._run.define_metric("train/*", step_metric="update")
            self._run.define_metric("dev/*", step_metric="update")
            self._run.define_metric("benchmark/*", step_metric="update")
            self.record.update(status="running", run_id=self._run.id,
                               run_url=self._run.url,
                               project_url=self._run.get_project_url())
        self._call("initialization", initialize)

    def log(self, metrics, *, step=None):
        def write():
            if self._run is None:
                raise RuntimeError("W&B run has not started")
            self._run.log(dict(metrics), step=step)
        self._call("metric logging", write)

    def summary(self, metrics):
        def write():
            if self._run is None:
                raise RuntimeError("W&B run has not started")
            self._run.summary.update(dict(metrics))
        self._call("summary logging", write)

    def finish(self, *, succeeded):
        if self._run is None:
            return
        already_failed = self.record["status"] == "failed"
        self._call("final synchronization", lambda: self._run.finish(exit_code=0 if succeeded else 1))
        if not already_failed:
            self.record["status"] = "synced" if succeeded else "synced_failed_experiment"
