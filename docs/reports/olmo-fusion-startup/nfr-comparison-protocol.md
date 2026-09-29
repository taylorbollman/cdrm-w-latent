# Endpoint-only comparison of the conditional NFR continuations

This CPU-only analysis compares the two authorized update 4→20 continuations.
Both start from the **same BF16-trained update-4 model, Adam state and RNG**.
It is conditional on that common history; it does not compare independent
FP32 and BF16 training from initialization. No additional model execution,
training, precision variant or numerical acceptance threshold is introduced.

Use the original four-update report and the completed FP32/BF16 continuation
reports, each with an independently supplied SHA256. Require all original
source pins, fixed NFR mode, data/fixture authority, update indices 148–163,
actual token counters, LR schedule and common initial boundary. The only
allowed initial state change is the already tested scheduler-prefix extension
from 4 to 20. Require common-FP32 held-out evaluations at 4/12/20, exact initial
aggregate evaluation, read-only checks and per-pass count/weight reconciliation.
CE weights are 1/2, 1/6, 1/6, 1/6; latent/KL weights are 1/4 each. No failure can be
discarded merely because its metrics appear favorable.

Read exactly three local full checkpoints: the retained common BF16 update 4,
FP32 update 20 and BF16 update 20. Each must have a verified whole-byte GCS
receipt. Verify its local SHA256 once, read it with `weights_only=True` and
CPU memory mapping, then check the complete serialized boundary against its
report, configuration/source authority, canonical parameter ownership, tied
aliases, Adam steps and finite nonnegative second moments. Recheck file stat
signatures after analysis to catch mutation of memory-mapped backing files.
No fresh cloud download is needed; the producer has already verified retained
checkpoint bytes. Three checkpoints total about 45.6 GB of local serialized
data, but no full FP64 model or per-step vector histories are allocated.

Reuse `gradient_geometry` for model parameters and Adam first/second moments,
grouped as backbone, fusion, predictor and all. In every comparison FP32 is
the reference and BF16 the actual endpoint. Also subtract the common origin
in FP64 one parameter at a time to measure actual cumulative parameter and
moment changes, including their absolute norms, difference norms, relative
errors and cosine similarities. Tied parameters count once. A small difference
relative to the entire pretrained weight vector cannot replace the cumulative
update comparison. Scalar origin norms provide scale without repeated
self-comparisons over full tensors.

Produce two PNG/PDF figures: per-term training trajectories beside common-FP32
held-out trajectories, and the four unweighted held-out pass losses at 4/12/20.
Label the shared BF16 origin and each training precision. Training rows use
different batches and are not themselves evidence of held-out improvement.
Latent/KL means and CE nats per target are reported separately so a combined
objective cannot hide opposing changes.

The new helper is `scripts/olmo_fusion_startup_nfr_compare.py`; its focused CPU
tests exercise literal concatenated-vector geometry, delta denominators, tied
ownership, actual checkpoint bytes and boundary pins, invalid reports and
per-pass plot inputs. Freeze this helper, tests and protocol after review.
Run only when root authorizes after both endpoint reports and receipts are
complete, in the project container with `CDRM_DOCKER_GPUS=none`, initially with
a 900 s outer bound. Write atomic progress before each checkpoint load and after
each of six geometry reductions. Retain scalar evidence, source snapshots and
plots through the existing retention path. No model checkpoint rewrite, source
change, GPU use, quality conclusion or BF16 production clearance is implied.
