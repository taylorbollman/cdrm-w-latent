#!/usr/bin/env python3
"""Host orchestration only: every model command runs in the required container.

Run calibration, training, and final evaluation as separate reviewable phases.
Completed artifacts are synchronized to the user's specified GCS prefix.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibrate", "train", "evaluate"), required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if Path.cwd() != project:
        raise RuntimeError("Start the controller from the host project directory")
    plan = json.loads(args.plan.read_text())
    prefix = plan["storage_prefix"].rstrip("/")
    if not prefix.startswith("gs://fast-chunks/cdrm-w-latent/stage-b/"):
        raise ValueError("Unexpected storage destination")
    controller = args.run_root / "execution"
    controller.mkdir(parents=True, exist_ok=True)
    budget_file = controller / "budget.json"
    if budget_file.exists():
        budget = json.loads(budget_file.read_text())
    else:
        budget = {"started_unix": time.time(), "wall_budget_seconds": 7200,
                  "scope": "Calibration, SYN training, final evaluation and intervening uploads; CPU preparation excluded"}
        budget_file.write_text(json.dumps(budget, indent=2) + "\n")
    deadline = budget["started_unix"] + budget["wall_budget_seconds"]
    history_path = controller / f"{args.phase}.json"
    if history_path.exists():
        raise FileExistsError("This phase already has an execution ledger; preserve it and resume individual jobs explicitly")
    history = {"phase": args.phase, "storage_prefix": prefix, "jobs": [], "status": "running"}

    def record():
        temporary = history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(history, indent=2) + "\n")
        temporary.replace(history_path)

    def synchronize(local, remote_suffix, log_path):
        command = ["gcloud", "--quiet", "storage", "rsync", "--recursive", str(local), prefix + "/" + remote_suffix]
        with log_path.open("a") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)

    record()
    try:
        for task in plan["tasks"]:
            for topology in ("seq", "r3"):
                if time.time() >= deadline:
                    raise RuntimeError("The bounded execution window has ended; completed jobs and checkpoints are retained")
                run_name = f"SYN-{task}-{topology.upper()}-seed{plan['seed']}"
                if args.phase == "calibrate":
                    run_name = run_name.replace("SYN-", "OPS-", 1)
                category = "calibration" if args.phase == "calibrate" else "runs"
                run_dir = args.run_root / category / run_name
                command = ["python", "scripts/stage_b_train.py", "--plan", str(args.plan),
                           "--task", task, "--topology", topology,
                           "--fixtures-dir", str(args.run_root / "fixtures")]
                if args.phase == "calibrate":
                    command += ["--fixed-batch", "--output-dir", str(run_dir)]
                elif args.phase == "train":
                    command += ["--output-dir", str(run_dir)]
                else:
                    checkpoint = run_dir / (run_name + "-final.pt")
                    command += ["--output-dir", str(run_dir / "evaluation-final"),
                                "--evaluate-only", str(checkpoint), "--split", "test"]
                inner = ('test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && '
                         'nvidia-smi -L && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ' + shlex.join(command))
                launcher = ["bash", "scripts/docker_shell.sh", "bash", "-lc", inner]
                log_path = controller / f"{args.phase}-{task}-{topology}.log"
                entry = {"task": task, "topology": topology, "run_dir": str(run_dir),
                         "command": launcher, "log": str(log_path), "started_unix": time.time(), "status": "running"}
                history["jobs"].append(entry)
                record()
                print(f"START {args.phase} {task} {topology}; log={log_path}", flush=True)
                with log_path.open("x") as log:
                    process = subprocess.Popen(launcher, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    entry["host_process_id"] = process.pid
                    record()
                    while process.poll() is None:
                        if time.time() >= deadline:
                            # Only this controller's process group is signaled.
                            # Docker forwards the signal to the runner, which saves
                            # recovery state at a completed optimizer-update boundary.
                            os.killpg(process.pid, signal.SIGINT)
                            process.wait(timeout=120)
                            entry["status"] = "execution_budget_reached"
                            raise RuntimeError("Execution budget reached; inspect retained recovery checkpoint")
                        time.sleep(1)
                entry["exit_code"] = process.returncode
                if process.returncode:
                    raise RuntimeError(f"Job failed with exit {process.returncode}: {log_path}")
                if args.phase != "evaluate":
                    manifest = json.loads((run_dir / "manifest.json").read_text())
                    if manifest["status"] != "complete":
                        raise RuntimeError(f"Run is incomplete: {run_dir}")
                    entry["completed_updates"] = manifest["completed_updates"]
                    if args.phase == "calibrate":
                        initial = json.loads((run_dir / "evaluations/dev-u0000-metrics.json").read_text())
                        final = json.loads((run_dir / f"evaluations/dev-u{manifest['completed_updates']:04d}-metrics.json").read_text())
                        ratio = final["primary_macro_answer_ce"] / initial["primary_macro_answer_ce"]
                        entry["fixed_batch_ce_ratio"] = ratio
                        # An operational learnability gate, not a held-out metric.
                        if ratio >= 0.5:
                            raise RuntimeError(f"Fixed-batch CE did not halve for {task}/{topology}; inspect calibration")
                entry["model_execution_seconds"] = time.time() - entry["started_unix"]
                synchronize(run_dir, f"{category}/{run_name}", controller / f"upload-{run_name}.log")
                entry.update(status="complete_and_uploaded", gcs_uri=f"{prefix}/{category}/{run_name}",
                             finished_unix=time.time())
                record()
                print(f"DONE {args.phase} {task} {topology}; artifacts={entry['gcs_uri']}", flush=True)
        history["status"] = "complete"
    except BaseException as error:
        history.update(status="stopped", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record()
        # Keep controller logs and the exact execution commands durable too.
        synchronize(controller, "execution", args.run_root / "controller-upload.log")
    print(f"PHASE COMPLETE {args.phase}", flush=True)


if __name__ == "__main__":
    main()
