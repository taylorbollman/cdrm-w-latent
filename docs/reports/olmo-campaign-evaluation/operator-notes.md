# Evaluation-enabled execution

Use `python -m scripts.olmo_campaign_eval_execute` inside the project container,
launched by two-rank `torchrun`, with the same explicit NCCL environment and
bounded external timeout as the accepted shared runner. The new CLI requires
a pinned declaration; native runs also require its independently pinned CPU
resolution. The old launcher remains frozen and evaluation-deferred.

The example commands and declarations are in
`.runtime/olmo-campaign-evaluation/`. `launch.py` checks every frozen source
before launching. Its `declarations-02` passed the real CLI's CPU authority
preflight for tiny reference, tiny evaluation and native evaluation. The earlier
`declarations-01` is superseded preflight evidence: its storage prefix lay
outside the historical retainer's supported root. No GPU run used it; the
corrected declaration keeps the retainer unchanged.

The native example is only a three-update NFR acceptance plan. It is not a
production budget. Resume restores the exact declared model, Adam, schedule,
counters, RNG and cursor; it cannot extend this immutable plan or import a
checkpoint from the older evaluation-deferred identity. The previous completed
run is an offline parity reference, never a resume prerequisite.

Evaluation uses a separately pinned dev index and a fixed prefix. Each pass has
its own `dev/pass_N/ce`, `latent`, `kl` and target-count W&B metrics. Disabled
auxiliary means are omitted. Counts are reduced globally before division;
`dev/aggregate/objective` uses the existing campaign pass weights and auxiliary
weights. This teacher-forced FP32/no-jitter evaluation is a common measuring
path; training still uses its declared BF16/Flash/native-Triton path and jitter.

Fresh evaluation and training metrics at the same optimizer step are merged
into one W&B log call. A restored eligible evaluation boundary logs once in its
new segment before the next training update. This prevents duplicate explicit
W&B steps from discarding training or evaluation metrics.

Only successful evaluation is published. If preservation or execution fails,
restart from the previous committed checkpoint; do not salvage the unverified
live state. Evaluation and heavy acceptance hashing contribute to wall time.
The checkpoint interval is checked at completed boundaries, so a long single
evaluation or transfer may exceed it. Reports distinguish forward/evaluation,
graph preparation, observation, serialization and retention costs.

Retained evidence uses the standard small-stage root under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T155500Z/`.
Checkpoint declarations use the historical retainer-compatible
`olmo-fusion-startup/20260929T161500Z/campaign-evaluation/` subtree. This storage
name does not change the explicitly recorded execution/startup identity.
