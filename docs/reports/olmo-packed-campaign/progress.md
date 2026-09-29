# Packed campaign readiness progress

2026-09-29: started `feat/olmo-packed-campaign` from `7b9c614`. Two idle H10080GB
GPUs verified inside the required container; retained Dolma token manifest is
present on SSD. No retokenization or quality run. Frozen policy/protocol in
`protocol.md`; runtime/model policy, disk-backed data and bounded component
precision probe are being implemented independently. Root owns GPU launches
and actual-data training/restart integration. Preserve progress every 20–30 min.

2026-09-29 resume: no old GPU processes were left running. Explicit stream
policy/model/loss changes and bounded per-component precision probe are committed
in `b9985bc`. Disk-backed index and real-data restart runner are implemented.
Independent model, data and runner reviews found no blocking issue; a narrow
index-file verification race is being hardened before runtime freeze. Initial
scoped CPU passes: model 378 plus one focused boundary case, data 18, precision
13, runner/probe integration 33 (overlapping scopes). Combined regression is
running; first launch only failed collection due a nonexistent test glob and
ran zero tests. No new GPU acceptance claim yet.

Evidence root: `.runtime/olmo-packed-campaign/`. New cloud namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T023500Z/`.
Upcoming order: tiny packed eager/graph, pretrained packed prepared reference,
bounded isolated CE/latent/KL precision localization, actual packed T1024 write,
cloud retain/restore, then fresh-process resident-Adam cold capture/resume.
The real update has 524,288 valid inputs, 512 chunks; B12/rank on two GPUs needs
22 accumulation slots/rank with padded final slots. All qualification failures
remain separate from operational readiness. No quality training queued.

Runtime/data freeze `f0be95d` pushed. Broad CPU regression: 971 passed in
75.32s; final data hardening scope: 20 passed in 3.40s (overlap). Tiny two-GPU
eager and CUDA graph probes both pass all eight arms under stream semantics,
including independent gradients and three-update Adam parity. Their reports
are retained (eager verified, graph upload pending at this entry). Actual
pretrained short graph probe is running; source hashes must stay unchanged.

Actual pretrained packed B/NFR graph probe passes operationally; independent
BF16 NFR sparse/dense qualification remains failed and is retained separately.
Actual index audit passes all 6,947,277 train tokens / 6,785 chunks against an
independent document iterator. Manifest hash
`372a7e05f5164198bdeb1531fc45bd761f5d33222d424c6804b54f19fd75288d`.
Train selected docs: 12,283; CE 6,940,492; latent 6,928,229; KL 6,909,206.
Finite schedule is 13 full logical updates plus 131,533 tokens (14th update),
no cycling. First two updates each end with rank1 entirely empty and rank0
eight real rows; actual-mask accounting passes. Index cloud retention verified;
restoration to a different path underway. Bounded per-loss precision diagnostic
running before the full-length checkpoint probe.

Bounded component diagnostic completed operationally: 13/13 health/state gates,
12 backwards, no optimizer. It reproduces the isolated 3.40224% BF16 layout
difference, with CE-only layout gradients exact and auxiliary-only discrepancies
~2.17% latent / ~2.28% KL. More significant: common BF16/full-FP32 combined
gradient error is ~85.96–86.13%, cosine ~0.51, BF16 norm ~0.495x FP32 on the
short initial NFR fixture. This is a new material qualification, not numerical
clearance. No root cause or harmlessness claim. Finish authorized recovery
mechanics; recommend numerical localization before longer training.

Actual T1024 write is running in `.runtime/olmo-packed-campaign/pretrained-write-01`;
checkpoint target `/mnt/localssd/cdrm-checkpoints/packed-campaign/pretrained-write-01`.
W&B `8c13f3ne`. Initial preparation starts without Adam; fresh-process resume
will load real Adam first. Index cloud restore passed at
`.runtime/olmo-packed-campaign/index-cloud-restore-01/evidence/index`; exact
generations, downloaded SHA/bytes and all 14 archive members verified.

Actual T1024 first logical update passes all five raw/state/finite/RNG-cursor/count
gates. Measured 3,968.23 valid input tokens/s globally: 7.823s loader+jitter,
123.702s graph backward, 0.597s Adam+cursor; diagnostic hashes/gates excluded.
Peak reserved 58.8984GiB/GPU, sampled free 14.3966GiB. Checkpoint saving at
~603s into write phase, then record live-graph next update. Cold Adam-before-DDP
restart remains untested at this entry.

Checkpoint after actual update one is committed. To reduce interruption
exposure, a separate immutable `checkpoint-boundary-01` evidence stage is
uploading its full checkpoint while the original write process records update
two. Retention receipt target: `retention/checkpoint-boundary-01.json`, GCS
stage `packed-checkpoint-boundary-01`. The completed write report will later
be retained separately without uploading a duplicate 15GB state. Restore helper
is prepared under `checkpoint-restore-evidence-01/restore_checkpoint.py`; use
that boundary receipt once verified. Do not resume until the original writer
finishes and its completed report hash is pinned.
