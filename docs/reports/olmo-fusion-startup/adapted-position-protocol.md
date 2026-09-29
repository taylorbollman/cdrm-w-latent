# Supplemental adapted-state position replay

2026-09-29. This bounded supplement uses the already implemented startup probe
to locate the retained adapted O5c hidden-state discrepancy. It adds **two
aggregate CE cases, four physical backwards and zero optimizer updates**. It
does not alter the startup-training protocol, model, loss, importer or historical
helpers. Root may schedule it independently on GPU1 while GPU0 performs startup
preflight/training; overlapping timings are not throughput measurements.

Run `scripts/olmo_fusion_startup_probe.py` with `--state adapted-o5c`,
`--fixtures original`, the pinned O5c `update-000512.pt` and its expected SHA.
Use a new immutable output stage, online W&B and a 900-second GPU phase bound.
Retain the finalized report, source snapshot, launcher log and this protocol.
There is no new checkpoint, fresh-fixture run or additional training in this
supplement.

## Authority and unchanged execution

The prior adapted report is
`.runtime/olmo-adapted-precision/adapted-01/report.json`, SHA256
`6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4`.
The historical O5c report is pinned at
`020a204ae02d3dfa1af2753f278cb465d73120ca8133258bc8a84272e15afbc8`.
The imported checkpoint is 4,807,843,871 bytes, SHA256
`7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`,
retained as generation `1790059437165208` at
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5c-fusion-only/20260922T061000Z/mixed/update-000512.pt`.

Read-only review confirmed that the existing CLI requires a saved-state SHA;
the importer validates the independent endpoint authority, full checkpoint
hash/size, unchanged file identity during loading, historical source mapping,
tensor keys/shapes/dtypes/finiteness, saved native/buffer pins and complete fusion
state including output scale before copying weights. It preserves parameter
ownership, training modes, full FP32-master trainability, tied readout and the
fresh predictor. It restores no optimizer, scheduler, historical RNG or cursor.
Current loaded state must match the previous adapted report exactly. This is
current-runtime weights-only execution, not historical O5c training reproduction.

Use the existing NF K4/beta1/jitter0.02 setup, original T16/B2 fixture with two
physical records, 29 inputs and 25 CE/25 latent/21 KL eligible positions. The
CE objective retains campaign pass weights; auxiliary cotangents remain zero.
Compare full FP32 math/eager with BF16 Flash/Triton at identical saved weights,
tokens, masks and keyed noise. Deterministic setup precedes CUDA, TF32 stays off,
and there are no temporal RT layers, DDP, CUDA graphs or optimizer.

## Required replay check before interpretation

The probe's runtime first-pass identity check is necessary but insufficient for
this supplement. Independent post-run audit must compare both precision rows
against the retained adapted report using the existing anchor-comparison
semantics: exact loss/count metrics, **all-pass** forward fingerprints, gradient
group summaries, full forward/cotangent comparison geometry, and BF16 group and
per-parameter gradient comparison summaries. Also verify fixture/state pins,
import checks, source snapshots and final state/RNG integrity.

Old full gradient vectors were not saved. Exact saved summaries and forward
hashes therefore establish the available replay scope; do not describe this as
cross-run gradient-vector byte identity. If an anchor differs, report that
limitation before interpreting positions as the source of the old discrepancy;
do not silently substitute a nearby run.

The retained target is BF16 versus FP32 hidden relative L2 **12.4366% at physical
record 0, pass index 1**. Retained backbone/fusion gradient relative L2 values
are 0.908519%/1.405077%. The new position observations should report the largest
individual errors with absolute/reference norms, valid-token and direct-CE
flags, feedback source/destination eligibility, and the actual/reference
incoming-cotangent support. Compare the common all-valid and union-supported
aggregates, including any positions with exactly zero cotangent in both cases.

Support is descriptive and never filters the actual backward. Missing direct
CE does not imply absence of indirect loss influence; zero observed cotangent
does not prove harmlessness. This replay can identify where the old spike
occurs. It cannot establish its cause, bless BF16, show training quality or
generalize to NextLat auxiliary losses, RT, longer sequences or distributed
training.
