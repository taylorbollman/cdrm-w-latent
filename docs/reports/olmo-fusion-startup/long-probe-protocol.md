# Supplementary T128 fusion-startup precision observations

Authorized 2026-09-29 within the overnight window. This is a separate bounded
supplement to the two short fixtures, with unchanged model mathematics, losses,
checkpoint import, numerical budgets and previous frozen evidence. Freeze this
file, new helper/tests and imported source inventory before either GPU stage.

## Data frozen before outcomes

Verify the prepared corpus manifest
`f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76`
and the existing T128/B8 startup training manifest
`2316978559b8db357c8d4adf706e41c94f809922171e8fb0ba50dc17ad66cbc0`.
Exclude the four development documents in the existing short fixture, pinned by
`5e55ee7bab67bcffb9fb01de48b9a971bae6ecbee59b8b7ebdf189c52600918f`.

Among remaining dev documents with at least128 real tokens, select four by
round-robin alphabetically sorted source names and ascending document keys within
each source. This improves source coverage without selecting by observed model
results. Record eligible source counts and complete chosen source/document pins.
Do not use the training or reserved confirmation splits. Duplicate-content and
split checks remain those of the verified prepared corpus/startup data helper.

Take each chosen document's actual first128 tokens. Preserve internal or terminal
EOS exactly; do not fabricate a prefix EOS. No packing or padding. Two physical
B2/T128 records contain512 inputs,508 CE targets,508 latent pairs and504 KL
triples. All eligible within-document target masks are enabled; first target slot
is false. Use the existing row-keyed jitter generator, with distinct long-fixture
keys, current recipe seed, logical update0, K4 and FP32 noise. Export bounded JSON
with tensor values, tensor hashes and regenerated noise hashes; freeze its SHA
before GPU execution. Pure CPU export/load must round-trip exactly and preserve
caller RNG. Old data helpers and short fixtures remain unchanged.

## Exactly two saved states

Run cold original OLMo step60000 plus fresh fusion, then the saved update128
fusion-startup endpoint. Each state performs the established full-trainability
NF FP32/math and BF16/Flash comparison: two aggregate cases/four physical
backwards. Across both states the total is four aggregate/eight physical
backwards. No optimizer construction, step, training continuation, extra dtype
or kernel sweep is included. No intermediate32-state long probe is queued.

Retain K4, beta1, jitter0.02, campaign CE pass weights(1/2,1/6,1/6,1/6), one shared
global target denominator and active predictor/latent/KL branches with zero
auxiliary cotangents. FP32 master parameters, ordinary checkpointing, native
RoPE, eager pointwise, deterministic controls beforeCUDA, TF32off and autocast
cache off match the frozen helper. Production flags are restored between cases.
The probe recipe remains T1024 while the physical diagnostic is T128.

Cold model state must equal the retained NF matrix origin. At128 import only
complete fusion state through the frozen compact loader; verify frozen native,
predictor and output-scale authority, update128 and matching data manifest.
Compare first-pass hidden/embedding fingerprints exactly with this new cold
fixture at the same precision. Later hidden states/cotangents need not match the
cold model. Exact source/runtime/reduction/fixture controls keep both states
comparable; no prior short-fixture forward fingerprints apply to these new data.

## Observations and limits

Reuse measure_pair() and existing operational health/state/RNG checks. Report
CE and descriptive full/backbone/fusion/predictor raw-gradient geometry, each
pass's hidden/full incoming-cotangent differences, and per-valid-position
geometry. Preserve direct CE and feedback masks and union-of-both-precisions
cotangent support summaries. A missing direct target or zero observed cotangent
does not establish harmlessness. These observations add four documents and
training-length contexts; they are not production-scale coverage or a revised
BF16 acceptance threshold.

Root alone launches GPU stages inside the project container, each under an
external900-second cap. OnlineW&B, atomic per-case reports, source snapshots and
immutableGCS evidence are required. CPU preparation is separate and GPU-free.
No original source file is changed. Stop after the cold/update128 pair of pairs
and assess with the short-fixture results; optimizer/update qualification is a
different explicitly scoped follow-up.
