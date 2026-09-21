# OLMo O1: frozen bounded validation protocol

Protocol recorded 2026-09-21 before actual-checkpoint validation completed.
This milestone implements ordinary native fidelity and a sequential RT
reference. It performs no model adaptation or comparative learning run.

## Artifact and source

- Original `allenai/OLMo-1B@81b71efbce6f4dada57c94860301af4298bcd351`,
  `step60000-tokens252B`; approximate published age 251–252B.
- Native tokenizer from the same revision, no automatic BOS/EOS. Validation
  fixtures are two literal text/code strings with their exact token IDs saved
  in the report; model logits retain all 50,304 vocabulary rows.
- Native math executes pristine AllenAI source at
  `b3741bc21f1dd504838b7dbd9878ee077ded63bd` in an isolated namespace.
  The source manifest hashes original files; only import/framework plumbing is
  adapted, as described in the reference README.
- Adapter and reference load the same 65 tensors, totaling 1,176,764,416
  parameters. Native weight tying, LayerNorm, SwiGLU ordering and RoPE persist.
  The checkpoint is fully hashed before loading; no partial state import.

## Ordinary fidelity gate

Actual native size, batch 1, text and code fixtures capped at 64 tokens. Compare
outputs/loss and final hidden states on both fixtures; compare all parameter
and embedding-input gradients on the text fixture. Runtime choices:

1. FP32 parameters, autocast off, TF32 off, math SDPA.
2. FP32 parameters, BF16 autocast, math SDPA on both implementations.
3. FP32 parameters, BF16 autocast, default SDPA on both implementations.

Source-parity tolerances: FP32 outputs absolute/relative 3e-5; gradients
absolute 2e-6 / relative 3e-5. BF16 source outputs absolute/relative 2e-3; gradients
absolute 2e-4 / relative 3e-3. These compare native and adapter in the same precision;
they are not thresholds for BF16 differences from FP32. Cross-precision
differences are recorded descriptively, per tensor and globally.

Check full versus chunked adapter execution, native cached decoding, future
token isolation and native KV shapes. Observe actual ordinary BF16 SDPA dispatch
with a bounded profiler invocation. This is not a throughput benchmark.

## Sequential RT gate

Actual native size, batch 1, T16, selected layer 0 only. Four FP32 cases:

- Alpha 0 scan versus native ordinary computation.
- Alpha 0, 0.37 and 1 scans versus an independent history-reconstruction oracle.

The oracle uses original block primitives, reconstructs past input/output
mixtures at every position, and computes attention explicitly. It shares no
production scan/cache implementation. Compare all 65 parameter and input
gradients, logits, hidden states and CE. Alpha 0 follows the actual scan.

FP32 output tolerance absolute/relative 1e-4; loss absolute/relative 1e-5;
input gradients absolute 2e-6 / relative 3e-4. Each parameter tensor must satisfy
both relative L2<=1e-4 (or total absolute error norm<=2e-6 for nearly zero tensors)
and maximum error<=2e-6+1e-4*reference_max_abs. Retain stricter elementwise
diagnostics. These are initial semantic screens for reduction-order differences;
do not relax them silently after seeing results.

Bounded BF16 checks: alpha 0 versus native ordinary, alpha 1 versus oracle with
math and default SDPA. Require finite results and complete gradient ownership;
record differences from FP32 without treating those observations as BF16
training clearance. Test chunked caches and causality at all three alpha values.

Tiny CPU tests supplement actual-checkpoint coverage: layer placements,
fractional-alpha branches, attached cached training, multiple outstanding
shared-weight forwards, metadata copying, stale/mismatched cache rejection,
weight tying/materialization and a bounded optimizer/save-resume fixture.

## Evidence and limits

Each GPU driver creates a new output directory and logs online to
`taylorbollman/pretrained-fbt-rt-nextlat`. Save runtime, source hashes, expanded
native configuration, exact token fixtures, each tensor's errors, peak memory,
W&B URL and final status. Failures remain recorded separately from any corrected
rerun. The source checkpoint and passing evidence are retained on `gs://fast-chunks`.

This does not validate tiled execution, FBT, NextLat, long contexts, large
batches, distributed training, adaptation stability or downstream quality.
Memory usage is for paired validation models with gradients, not a batch-size
or training-capacity estimate.
