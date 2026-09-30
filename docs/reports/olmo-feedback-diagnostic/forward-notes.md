# Forward diagnostic interpretation

The full-strength feedback deficit is already present at the actual saved NF
origin. At update 32, bypassing feedback restores the checkpoint's own first-pass
behavior exactly, while the tested half-strength interpolation does not improve
later-pass CE. This localizes a problem in using the feedback route, but does
not distinguish inadequate adaptation from a conflicting training objective.

These observations use the same eight predeclared T1024 development rows:
8,192 inputs, 8,184 CE targets, 8,172 latent pairs and 8,152 KL triples. They
are a small diagnostic subset, not the cohort's 65,536-input development panel
or a final evaluation. Every case uses common FP32/no-jitter inference on copied
saved weights; no optimizer updates occur. NF origin includes the existing
fusion128 import and the actual saved fresh predictor.

| Checkpoint / beta | Pass 1 CE | Pass 2 CE | Pass 3 CE | Pass 4 CE |
| --- | ---: | ---: | ---: | ---: |
| NF origin / 1 | 2.67387 | 7.36410 | 7.15750 | 7.18838 |
| NF32 / 1 | 2.94735 | 7.34760 | 7.38958 | 7.40077 |
| NF32 / 0.5 | 2.94735 | 7.43257 | 7.43654 | 7.43946 |
| NF32 / 0 | 2.94735 | 2.94735 | 2.94735 | 2.94735 |
| NFR32 / 1 | 3.03240 | 6.97403 | 7.01015 | 7.01681 |
| NFR32 / 0 | 3.03240 | 3.03240 | 3.03240 | 3.03240 |

Relative to origin at beta 1, endpoint CE worsens by 0.27348 on pass 1, improves
by only 0.01650 on pass 2, and worsens by 0.23208/0.21240 on passes 3/4. This is
different from the cohort's observed improvement between updates 16 and 32;
those observations cover a different time interval and a larger panel. The
origin comparison rules out a claim that these 32 updates created the entire
later-pass deficit, but it does not establish steady improvement from origin.

For all eight NF32 rows, first-pass hidden hashes are identical across the three
beta settings. At beta 0, all four passes' hidden hashes, losses and actual
stack inputs agree exactly, fusion is never called, and raw-fusion statistics
are absent. These controls support the intended bypass and fresh-pass execution.
They do not constitute a general numerical equivalence result.

The completed NFR confirmation also passes all eight beta-zero controls, with
native RT retained at layers 0/15 on every pass and first-pass hashes unchanged
across beta settings. At full strength, its first-pass CE is 0.08505 worse than
NF32 and later-pass CE is 0.37358/0.37943/0.38396 better. Later passes still
trail NFR's own first pass by 3.94–3.98 nats/target. This reproduces the cohort's
mixed RT signal on the small subset; it does not establish useful refinement.
NFR beta 0.5 was not run because that control did not repair NF.

## Auxiliary losses and readout distributions

| Checkpoint / beta / pass | Latent mean | KL mean | Teacher entropy | Student entropy |
| --- | ---: | ---: | ---: | ---: |
| NF origin / 1 / 1 | 0.95821 | 9.86459 | 2.64145 | 2.45139 |
| NF origin / 1 / 2 | 0.79108 | 6.03154 | 6.92048 | 1.81690 |
| NF origin / 1 / 3 | 0.81578 | 6.40824 | 6.77782 | 2.18470 |
| NF origin / 1 / 4 | 0.81806 | 6.38076 | 6.78666 | 2.20770 |
| NF32 / 1 / 1 | 0.30493 | 2.71029 | 3.94736 | 4.69088 |
| NF32 / 1 / 2 | 0.03751 | 0.31703 | 7.66384 | 8.01280 |
| NF32 / 1 / 3 | 0.03753 | 0.32268 | 7.69164 | 8.02491 |
| NF32 / 1 / 4 | 0.03753 | 0.32383 | 7.69837 | 8.02819 |
| NFR32 / 1 / 1 | 0.29177 | 2.61159 | 3.88503 | 4.21302 |
| NFR32 / 1 / 2 | 0.04817 | 0.47085 | 6.85663 | 6.65346 |
| NFR32 / 1 / 3 | 0.04973 | 0.47529 | 6.77331 | 6.71070 |
| NFR32 / 1 / 4 | 0.04968 | 0.47425 | 6.78248 | 6.71724 |

At the endpoint, the later-pass predictor and teacher have much closer readout
distributions, and both distributions are broad. Their entropy increases while
the auxiliary losses fall and language CE remains poor. This is consistent with
learning an easier auxiliary prediction problem; it does not demonstrate useful
language refinement or establish that the auxiliary objective caused the deficit.
NFR's later readouts are less diffuse than NF's and have higher auxiliary losses,
alongside somewhat lower CE. Lower auxiliary loss is therefore not a reliable
ranking of language prediction even between these two endpoints.

Entropy is in nats on the exact KL-eligible triples, using all 50,304 vocabulary
rows. It is a diagnostic teacher/student recomputation, with possible small
rounding changes from regrouping predictor GEMMs; it is not an exact
reconstruction of logged KL or sampled-generation entropy. Entropy alone does
not measure task-relevant information or prove representation collapse.

## Hidden variation and feedback geometry

| Checkpoint / beta | Centered RMS, pass 1 | Pass 2 | Pass 3 | Pass 4 |
| --- | ---: | ---: | ---: | ---: |
| NF origin / 1 | 0.76544 | 0.20975 | 0.24755 | 0.25261 |
| NF32 / 1 | 0.49853 | 0.09162 | 0.09561 | 0.09716 |
| NF32 / 0.5 | 0.49853 | 0.06526 | 0.06406 | 0.06425 |
| NF32 / 0 | 0.49853 | 0.49853 | 0.49853 | 0.49853 |
| NFR32 / 1 | 0.49084 | 0.14650 | 0.14736 | 0.14714 |
| NFR32 / 0 | 0.49084 | 0.49084 | 0.49084 | 0.49084 |

Total hidden RMS stays approximately 1 because these are final-normalized states.
The table is the square root of the element-weighted mean **within-row centered
variance**: each B1 row has its own feature-wise positional mean removed before
combining squared deviations. It is neither the arithmetic mean of row RMS
values nor variance after subtracting one shared eight-row mean. Lower values
show reduced positional variation in this statistic, not proof of collapse.

Raw fusion RMS is approximately 0.037076 throughout, versus embedding RMS
0.039835 at origin and 0.039849 at the endpoint. The fixed output normalization
makes this similar scale unsurprising. At NF32 beta 1, pooled raw-fusion versus
embedding cosine is -0.00187/-0.00306/-0.00322 for passes 2/3/4, respectively;
the feedback vector is almost orthogonal to its corresponding token embedding
under this aggregate statistic. Similar RMS therefore does not mean a small
input perturbation.

At beta 0.5, actual executed input cosine to the embedding rises to
0.73153/0.73111/0.73106, while its RMS is only about 0.682 times embedding RMS.
The interpolation changes both direction and amplitude, yet does not repair CE
at this tested strength. This single control does not exclude benefit from
another strength or a trained interpolation schedule, neither of which was
tested. At beta 0, executed inputs are exactly the original embeddings.

NFR32 raw-fusion cosine is similarly near zero: -0.00355/-0.00358/-0.00359,
with RMS about 0.037076 against embedding RMS 0.039864. Its later-pass centered
RMS is higher than NF32's, but remains well below its own first pass. These
observations describe a different endpoint representation; they do not isolate
which RT-dependent computation produced the difference.

Fusion cosines pool raw dot products and squared sums over all 8,184 eligible
shifted positions and all features. They are not averages of token cosines or
cosines between pooled means. Continuous-stream feedback crosses true document
boundaries; the stricter KL/latent masks are used only for those losses.

## Evidence and aggregation

Loss means and entropy means above are recomputed from eight-row numerators and
their own denominators; they match the saved per-pass summaries. Model weights,
RNG and gradient buffers are unchanged in all three completed case reports, and
evaluation restoration checks pass. The report fixture membership is
`407798544e9372034f76d12e60cfe601ab667b0306309ea4623f0de47c1914df`.

The immutable case JSONs under `.runtime/olmo-feedback-diagnostic/` are:

| Case | SHA256 |
| --- | --- |
| `nf0-forward-02/beta-1.json` | `f5a5c4967fbae36465a940bf0c1ddbd1dec30470180826169685996805e4d617` |
| `nf32-forward-01/beta-1.json` | `ee457fd0b9d6badc6aba2f0fb07c4f0d882468c72c296718229ccacfe5b9a138` |
| `nf32-forward-01/beta-0.5.json` | `4e1ef98b345e8938fac47a1084d8a56dba5c2cebeb3a66f2aabb9eb31d6ad728` |
| `nf32-forward-01/beta-0.json` | `cbf882e2e1b799abe534562114bcb19a796cf5df221e6bd2a26d6f7fb2debea6` |
| `nfr32-forward-01/beta-1.json` | `7242cff5d714a2708bdaab3fa130188966bceca404a6b54e8e68c4cc598192e5` |
| `nfr32-forward-01/beta-0.json` | `01b10b3129a01b966ef9d88182f651600da68e1a3d6b1035c7aa8fca853bdeb0` |

The separate training-batch gradient diagnostic is not included in these
forward findings. Neither the forward controls nor the auxiliary
trends alone justify a loss-weight change, continuation or model-selection claim.
