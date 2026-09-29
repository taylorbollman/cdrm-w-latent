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

Write phase complete:13/13 gates pass, 835.55s; second-update rate3,957.59/s.
Completed report SHA
`960e65563ac4b66a51197546cb14e8eccd6d457bfdb32838c8136333d5c8ce38`.
Checkpoint manifest SHA
`9238b186a33c80be85aa18aec11cc2ea15f097e137be34eece4e978ca2886a50`.
Boundary retention receipt verified; full checkpoint now in GCS. CPU restoration
into `/mnt/localssd/cdrm-checkpoints/packed-campaign/cloud-restored-01` running.
Only after downloaded bytes verify, launch new torchrun resume with the above
report+manifest pins, cloud-restored index path and unchanged runtime sources.

Full checkpoint cloud download verified: state15,214,757,825 bytes plus manifest
58,065 bytes. Fresh-process resume launched with new cloud-restored index path
and restored checkpoint, source/config pins unchanged. W&B `igpkw3gd`; stage
`.runtime/olmo-packed-campaign/pretrained-resume-01`. Exact configuration and
restored boundary gates pass; actual Adam is resident before DDP construction.
Cold warmup is running. Remaining: capture, restored next update and bitwise
comparison, then retain final evidence/audit and close draft PR39. Main report
and next-steps.md prioritize numerical localization rather than quality training.

Cold Adam-resident setup/capture passes, preserving model/optimizer/RNG/clocks/
cursor exactly. Peak allocated42.8022GiB, peak reserved58.3496GiB and
14.9454GiB sampled free per GPU after capture. Resumed next update is running;
bitwise continuation gate remains pending. No B8 fallback needed.

IMPORTANT: original `pretrained-resume-01` FAILED final bitwise gate at663.64s;
8/9 gates pass. Pre-update and postprepare state/input/noise exact, all scalar
losses exact, counts/RNG/cursor exact. All4 predictor grads match; all65backbone
+2fusion rawgrad hashes differ; preclipnorm196.2971954 vs196.2953186 (does not
bound vector error). Failedattempt cloudreceipt `pretrained-resume-01.json`
verified/publishing. GPUprocess ended. No restart acceptance yet.

Root authorized adaptive bounded test within this milestone: missing global
deterministic setup is leading explanation; existing helperconfigure_determinism
isused by older distributed harness. Runneragent patches onlynewpackedrunner+
tests to setcontrols beforeCUDA andpinmetadata/helper. Precisionagent addsfixed
QKV Flashrepeatability probe; rootwillrunT16/T1024 off/on. Ifpositive, new
write02/resume02 withnewsourcepins/checkpointboundary02; preservefailed01.
Protocoladdendumrecords this; no thresholdrelaxation. DraftPR39 remainsdraft.

Adaptive Flash test confirms T1024 ordinary backward nondeterminism without
deterministic controls (maxcombinedQKVrelativeL2 3.91786e-6); T16 off is exact.
Deterministic mode givesbitwiseexact eager/captured repeats atbothlengths.
Fourstage reportscomplete; microharness2080c11. CombinedfocusedCPUchecks18pass
in2.24s. Newpackedrunner e5a593b setscontrols BEFORE CUDA andpinshelper+
determinism metadata in checkpoints. Source/protocol frozen fornewpair.

`pretrained-write-02` launched at~03:20UTC, 1200s externalbound, sameB12/T1024
actualindex/inputs, newcheckpoint
`/mnt/localssd/cdrm-checkpoints/packed-campaign/pretrained-write-02`. RootGPU
launcher session12644. Oncecheckpointcommits, retain separatelyas
`checkpoint-boundary-02` whilecontinuationruns, then cloudrestoreinnew
`cloud-restored-02` andfreshresume02 withnewreference/manifestpins. Original
failedpair01remainsretained. FourFlashstagesretentionunderway.

Resumed after interruption: the writer remained active and completed normally.
Deterministic write02 passes 13/13 gates and two actual updates in 965.54s.
Report SHA `a3c1b48913ecc95f087cc1a79ddc3c70e6ac0642e0f0e5ee0f3883b42b230754`;
checkpoint manifest SHA
`c1cdb79fc58b767160bf6466777b434665f4271f731e03698549b28d85e74ed1`.
W&B `k5ynlpin`. Completed small write evidence retained. Full boundary02 upload
is active (root session37366); after its receipt verifies, use prepared
`checkpoint-restore-evidence-02/restore_checkpoint.py` to download into
`/mnt/localssd/cdrm-checkpoints/packed-campaign/cloud-restored-02`, then launch
resume02 with the completed report/manifest hashes above and restored index.
All four Flash stage receipts verified. No new-process acceptance yet.
