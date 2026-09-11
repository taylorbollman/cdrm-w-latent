# A5 length follow-up: usage and handoff

Completed 2026-09-11. Read [the result](reports/rt-a5/length-followup/report.md)
and [the approved plan](rt-a5-length-followup-plan.md). The evaluation-only
milestone is complete. Do not rerun training or reopen a precision study by
default. RT was evaluated and reported first; limited SEQ checks followed.

The unchanged primary checkpoints are from
`.runtime/rt-a5/20260911T154748Z/train-{rt,seq}/checkpoints/step-010000.pt`.
Both models have two layers, D512/H8/GELU FFN2048 and 6,357,504 parameters;
both RT layers are tiled recurrent at rho1. All new forwards used FP32,
no autocast/TF32/compile/CUDA graphs. Final confirmation remains unevaluated.

New utilities do not modify the historical trainer, model, data or evaluator:

- `scripts/rt_a5_prefix_check.py`: checks a frozen endpoint and a 1,024-word
  OOD subset, preserving checkpoint/source/data/runtime identity, fixture,
  logits, per-position integer counts and unchanged model hashes. RT uses
  five forwards T36/12/13/14/16; SEQ three T14/12/13.
- `scripts/rt_a5_prefix_assess.py`: CPU-only assessment of saved logits and
  their original strict screen. Preserves the failure, checks complete
  correct/incorrect masks, and reports probability/loss changes and margins.
  No model inference. The saved RT assessment used the first version, which
  additionally required identical predictions and certified margins; the
  later SEQ version permits changes between wrong classes when every
  correctness indicator agrees. Each source version is retained with hashes.
- `scripts/rt_a5_length_report.py`: CPU-only generation of full E/A/M curves
  from original 5k/10k records. Produces CSV/counts/pointwise Wilson E/A
  intervals, PNG/PDF, Markdown and online W&B tables/plots. It accepts an
  original failed strict screen only with an explicitly linked assessment
  supporting the requested accuracy metrics; no failed screen becomes a pass.

GPU commands must use the project launcher and verify the container/GPU.
For a deliberately new RT prefix check:

```bash
bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && nvidia-smi -L && python -m scripts.rt_a5_prefix_check --architecture rt --run-dir .runtime/rt-a5/20260911T154748Z/train-rt --data-dir .runtime/rt-a5/20260911T154748Z/data --output-dir .runtime/rt-a5/NEW/rt-prefix --wandb-group NEW'
```

The strict check can intentionally exit nonzero while retaining complete
evidence. Inspect it before deciding whether a saved-output assessment is
appropriate. The assessment refuses a meaningful change to correctness.
For example, in the explicitly GPU-disabled container:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && python -m scripts.rt_a5_prefix_assess --prefix-report .runtime/rt-a5/NEW/rt-prefix/report.json --output-dir .runtime/rt-a5/NEW/rt-assessment --wandb-group NEW'
```

To regenerate a report without a model forward, use a fresh output path:

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && python -m scripts.rt_a5_length_report --architecture rt --run-dir .runtime/rt-a5/20260911T154748Z/train-rt --output-dir .runtime/rt-a5/NEW/rt-report --prefix-report .runtime/rt-a5/20260911T165258Z-length/rt-prefix/report.json --prefix-assessment .runtime/rt-a5/20260911T165258Z-length/rt-assessment/report.json --wandb-group NEW'
```

The current reporter supports both saved assessment versions. Never weaken
checkpoint source matching; historical source SHA remains
`6f9d55a957bf505aefa1a351c33f0bc76291ff63736bc28b1b618393f46ef6ea`.
The reporter is intentionally limited to the original complete 10k pilot's
5k/10k full OOD evaluations. Future continuation reports need explicit history
assembly and support for their new endpoints, not reuse under a misleading label.

New evidence is under `.runtime/rt-a5/20260911T165258Z-length/`, and published
reports under `docs/reports/rt-a5/length-followup/{rt,seq}/`.
`rt-prefix/` and `seq-prefix/` retain failed strict screens; `rt-assessment/`
and `seq-assessment/` retain separate, qualified metric acceptance. The
31 distinct focused tests cover the prefix checker and reporter. Initial
test output and subsequent fixture/assessment-guard reruns are retained.

The central result is the RT tradeoff: E12 improves from 98.25% to 99.17%
between 5k and 10k, while E13 falls from 90.08% to 79.87% and E14 falls from
41.44% to 21.52%.
Do not promise that additional training improves length generalization.
Keep the 10k endpoint primary, report the 5k diagnostic transparently, and
retain the final confirmation set for a later fixed comparison. SEQ's
observed zeros justify ending redundant cumulative length sweeps on the
same predictions, not terminating training merely because exactness is zero.

All graphable new runs use entity `taylorbollman`, project
`rt-a5-state-tracking`. Main graphs:
[RT](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/oewgafq1),
[SEQ](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/eayae4ny).
Retain evidence under
`gs://fast-chunks/cdrm-w-latent/rt-a5/20260911T165258Z-length/`, with the
archive and checksum receipt, referencing original retained data/checkpoints.
No commit, external PR or push was created for this milestone.
