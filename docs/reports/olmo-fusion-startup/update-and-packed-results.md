# Saved-Adam and packed-context checks after fusion startup

The update-128 fusion remains substantially less sensitive to BF16 on both
additional checks. On the held-out T128 fixture, the actual fusion Adam update
differs by **1.685%** between BF16 and FP32; after subtracting the same
zero-gradient Adam control, the difference is **4.917%**. On two genuinely
packed T1024 rows, backbone/fusion gradient differences fall from
**19.970%/25.406% before startup to5.209%/8.175% afterward**. Absolute errors
and direction agreement improve too.

These results support continuing the bounded BF16 investigation. They do not
establish that all remaining errors are harmless, and they do not set a new
acceptance threshold. The optimizer experiment updates only fusion; the packed
experiment uses CE with zero auxiliary cotangents and no temporal RT. Actual
NextLat and RT combinations are separate component checks. The next useful
calibration is one ordinary-stack N-only CE precision pair on these same packed
rows, described below.

## What the optimizer check measures

Each candidate starts from the **same saved update-128 fusion, Adam moments,
scheduler and RNG**, with original native weights, predictor and output scale
frozen. The held-out data are the existing four T128 development prefixes:
two B2 microbatches,512 inputs and508 CE targets, K4/beta1/jitter0.02. This is a
counterfactual held-out update, not continuation on the8,192-target training
schedule. The checkpoint is restored afterward.

There are exactly three Adam calls: explicit zero gradients; FP32/math CE;
and BF16/Flash CE. The latter two require four physical backwards in total.
Saved LR is1e-4, betas(.9,.95), epsilon1e-8, weight decay0.1 and clipping at1.
FP32/BF16 raw norms are1.271680/1.270731, so both are clipped, with scales
0.786361/0.786948. Actual stored FP32 master-weight differences are observed
in FP64. The same two fusion matrices,8,388,608 parameters, participate.

Every row below compares the BF16 candidate with the FP32 candidate. Relative
L2 is the difference norm divided by the FP32 reference norm.

| Quantity | Relative L2 | Absolute difference norm | FP32 reference norm | Cosine |
| --- | ---: | ---: | ---: | ---: |
| Raw fusion gradient | 2.7115% | 0.0344812 | 1.271680 | 0.9996324 |
| Clipped fusion gradient | 2.7115% | 0.0271145 | 0.999999 | 0.9996324 |
| Actual master-weight update | 1.6846% | 0.00160920 | 0.0955253 | 0.9998581 |
| Update minus common decay | 1.6843% | 0.00160920 | 0.0955384 | 0.9998581 |
| Update minus zero-gradient Adam control | 4.9170% | 0.00160920 | 0.0327270 | 0.9987905 |
| Adam first moment | 0.8911% | 0.00271145 | 0.304265 | 0.9999607 |
| Adam second moment | 0.1839% | 0.0000291888 | 0.0158710 | 0.9999983 |

The zero-gradient Adam update has norm0.0870709; the common decay-only delta
has norm0.000642100. Thus the actual update contains a substantial contribution
from existing optimizer history. Its1.685% discrepancy should not be the only
number used to assess the new gradient. Subtracting the identical control leaves
the same absolute difference but a smaller reference norm, producing4.917%.
That residual still has cosine0.99879. It is an informative control, not a linear
causal decomposition of Adam or proof of equivalent long-run optimization.
The check contains no full-backbone optimizer and no second consecutive BF16
update. See the [fixed protocol](update-probe-protocol.md).

## Packed T1024 NF results

The two B1/T1024 rows contain2,048 valid inputs,2,046 CE targets,2,040 eligible
latent pairs and2,032 eligible KL triples. They span eight actual documents
from C4 and Common Crawl, with six true document boundaries. All documents are
development-only and exclude the eight documents used by the short and T128
fixtures. Selection was fixed before model execution and did not use results.

CE predicts across all six internal document boundaries. Six cross-document
latent pairs and12 boundary-crossing KL triples are excluded. Attention and
FBT feedback continue across those boundaries. There is no padding, fabricated
EOS or cross-chunk target. The endpoint is imported under its saved isolated
contract, then undergoes a separately checked policy-only change to
`continuous-stream-v1`, as does the cold control. The state and parameter
ownership are preserved. See the [packed protocol](packed-probe-protocol.md).

Each row compares BF16 with the same state's FP32 reference, with all diagnostic
parameters trainable. NextLat branches execute but their latent/KL cotangents
are zero; the predictor gradient is exactly zero. There are two aggregate
precision cases and four physical backwards per state, with no optimizer,
DDP or CUDA graphs.

| State / group | Relative L2 | Absolute difference norm | FP32 reference norm | BF16/FP32 norm ratio | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold backbone | 19.9698% | 25.97885 | 130.09096 | 0.95687 | 0.980134 |
| Startup128 backbone | 5.2089% | 0.821889 | 15.77845 | 1.01612 | 0.998793 |
| Cold fusion | 25.4055% | 4.942294 | 19.45362 | 0.92287 | 0.968254 |
| Startup128 fusion | 8.1753% | 0.205037 | 2.508017 | 1.03935 | 0.997530 |

The remaining discrepancy is larger than on the isolated T128 fixture. This
is not a controlled length-only comparison: data, physical batch shape,
context length and document policy also differ. Nevertheless, the reduction
persists with longer actual packed input, smaller FP32 gradient norms and
substantially smaller absolute errors. It is not an improvement obtained merely
by enlarging the denominator.

Weighted CE means are6.785020/6.783491 nats per target at the cold FP32/BF16
states and5.246691/5.247585 at startup128. The latter precision difference is
+0.0008947 nats per target. These fixed-fixture CE observations are numerical
context, not an evaluation result or evidence of a language-model quality win.

## Where the remaining differences occur

Every physical record/pass has1,023 positions with a nonzero incoming
cotangent in either precision. The final position has zero cotangent in both.
There are8,184 supported position/pass observations across two records and four
passes. The summaries below concatenate those observations for norm geometry;
they do not average token-relative errors. Full incoming cotangents include
later feedback paths.

For boundary-adjacent observations, include the token immediately before and
after each of the six true document transitions, in all four passes:
48 observations. All are valid, CE-supervised and supported. Zero-based start
positions of the new documents are639 in C4 and68,520,555,612,896 in Common
Crawl. Feedback eligibility crosses all six boundaries as specified.

| Scope / state | Hidden relative L2 | Incoming-cotangent relative L2 | Cotangent absolute difference | Cotangent FP32 norm | Cotangent cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| All supported / cold | 4.6950% | 21.6438% | 0.0576607 | 0.266408 | 0.976940 |
| All supported / startup128 | 1.2541% | 9.8362% | 0.00326986 | 0.0332432 | 0.995151 |
| Boundary-adjacent / cold | 6.2187% | 33.4154% | 0.00101943 | 0.00305078 | 0.942710 |
| Boundary-adjacent / startup128 | 1.2353% | 1.1372% | 0.0000206537 | 0.00181611 | 0.999936 |

Across individual record/pass *aggregates*, maximum supported hidden error
falls from12.120% to1.524%; maximum supported cotangent error falls from522.855%
to11.039%. Those maxima are not maximum individual-token errors. The remaining
11.039% aggregate occurs at C4 pass0, whose forward states are unchanged by the
learned fusion but whose incoming gradient includes subsequent feedback.

Among the48 boundary-adjacent individual positions, startup128's maximum
hidden error is2.460% and maximum cotangent error3.150%. The latter is C4
pass0/position638, with absolute difference0.0000116999, FP32 norm0.000371390
and cosine0.9995085. These measurements do not suggest a boundary-specific
precision failure in this fixture; they cannot rule one out on other data.
Supported-position filtering is descriptive and changes neither computation
nor an acceptance budget.

## Integrity and practical interpretation

The independent stdlib audit passes**111 checks**, rehashes the saved compact
checkpoint and both fixture files, and verifies all124/127/127 source snapshots
for update/cold-packed/startup-packed reports. It independently reconstructs
packed tensor byte hashes, document boundaries, slice hashes and target counts;
compares the recorded keyed-noise pins; checks both precision-specific pass0
fingerprints, unchanged native/predictor/output-scale pins, imported complete
fusion and policy-only state preservation; and checks the identical saved Adam
boundary and final restoration. It does not rerun GPU calculations, regenerate
jitter, reread the full native checkpoint, or repeat cloud downloads.

The audit and compact derived observations are retained in
`.runtime/olmo-fusion-startup/update-packed-audit-01/`. Audit report SHA256:
`bac499e289851030d982a49d955a5d4360ab9eabd2344683e6bbd043017735fc`.
Input report SHA256 pins are:

- Update128: `f9f0aa63c0ca2c2945e78e5fea62f1db6c91ed9fa16c84b60087ca593f0c6ca7`.
- Packed cold: `c1cfb4a0af1524872033828e8a802d26fb534543331785d11f30b42e9726ae02`.
- Packed128: `2b8311a50639428b4ce9ed390a260ab1cfd0e6bf66202d277cdbc477db65e947`.

The saved checkpoint is the same103,240,258-byte fusion/Adam endpoint reported
in [warmup results](warmup-results.md), SHA256
`892ff2fdcdeec89e3008a16a12e91158250ebe05adfe0e9efce8f153409b8cfc`.
Packed fixture SHA256:
`4932410f9fd370d9dae20a1075bf2a191c642e1563eb8557a6fe02fd7e83a975`.

A single N-only CE FP32/BF16 pair on the same packed fixture is useful next.
Keep the original pretrained backbone and predictor, disable FBT and RT, and
retain zero auxiliary cotangents. Pin each precision's forward to the retained
cold NF pass0. This calibrates the ordinary stack's discrepancy under the
actual packed input and dispatch settings; it avoids borrowing a short-fixture
number. There is no reason to run both cold and startup fusion states when
fusion is inactive. Compare within-arm numerical geometry, not raw gradient
norms causally across K1 and K4 objectives. This calibration and the separate
active-auxiliary/RT component checks should guide whether a bounded BF16
training continuation is justified. No production clearance follows from this
report alone.
