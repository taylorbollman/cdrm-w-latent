# Conditional NFR comparison reporting

This document records the predeclared reporting procedure; current execution
status is in [progress.md](progress.md).
Do not publish until both NFR branches have completed updates 33–64, retained
their update-64 checkpoints, and passed the independently pinned NFR pair audit.
The original 215-source execution authority and its protocol remain unchanged.

The CPU summary takes SHA-pinned parent32, control64, reduced64 and pair-audit
reports. It checks that the audit binds these exact reports and that both arms
are native NFR with RT at layers 0/15, four FBT passes, inherited original Adam
and unchanged 128-update schedule. It requires the same restored raw per-pass
evaluation at 32, data exposure, LR and dev membership in the pair. Missing
history segments fail; an interrupted/resumed pair needs a separately reviewed
explicit history join before using this reducer.

The report contains per-pass CE, raw latent and KL at 32/48/64; their separate
denominators; later-minus-first CE gaps alongside absolute CE; gradient norms
and clipping; parameter counts; and timings with clear regions. Rank timing uses
the slower rank per update, never a rank sum or an unweighted mean of rates.
Memory is a per-GPU maximum/minimum over samples, not a sum across GPUs.
Weighted objective totals are checked for bookkeeping and then excluded from
the summary, comparison tables, CSVs and plots.

Optional F-only context requires **both** a pinned immutable update-64 evaluation
and its update-64 publication receipt. The model must be F-only, the measurement
must use the identical development prefix and common FP32/no-jitter policy, and
only CE is shown. F128 is rejected. F64 is descriptive context, not a paired
intervention: its architecture, training losses and optimizer trajectory differ.

Reuse is limited to generic helpers in `olmo_kl_continuation_summary.py`: raw
loss/denominator checks, common four-pass evaluation, parameter accounting, CSVs,
timing definitions and applicable qualifications. Its NF-only branch guard and
plot labels are not reused or patched. New plots explicitly say NFR.

After completion and audit, run inside the project CPU container:

```bash
python -m scripts.olmo_nfr_kl_summary \
  --parent /path/to/parent32/report.json PARENT_SHA256 \
  --control /path/to/control64/report.json CONTROL_SHA256 \
  --reduced /path/to/reduced64/report.json REDUCED_SHA256 \
  --audit /path/to/pair-audit/report.json AUDIT_SHA256 \
  --output .runtime/olmo-nfr-kl-continuation/summary-01
```

Optionally add `--f64-evaluation JSON SHA256 --f64-publication JSON SHA256`.
The output is a fresh directory with all four/six input snapshots, analysis
source snapshots, JSON, three CSVs and four PDF/PNG figure pairs. This command
imports no Torch/model code and has no network or W&B side effects.

Publishing is a separate explicit command after inspecting the completed summary:

```bash
python -m scripts.olmo_nfr_kl_tracking \
  --summary .runtime/olmo-nfr-kl-continuation/summary-01/report.json \
  --summary-sha256 SUMMARY_SHA256 \
  --output .runtime/olmo-nfr-kl-continuation/summary-wandb-01
```

It verifies the summary and artifact pins, then publishes four figures and the
per-pass raw-loss table to `taylorbollman/pretrained-fbt-rt-nextlat`, in group
`nfr-kl-paired-continuation`. It does not resume training, change checkpoints or
rewrite either branch's history. Existing mixed-precision qualifications remain.
