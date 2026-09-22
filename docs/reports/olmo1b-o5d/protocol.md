# O5d: does fusion repair transfer to exact sequential feedback?

Authorized 2026-09-22 after O5c; the user permits several hours if needed.
This is evaluation only. No optimizer, training, architecture change, RT or
NextLat is introduced. The fixed grid is declared before evaluation.

## Checkpoints and execution

- Source: completed O5b FBT update2634, SHA256
  `99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66`.
- Repaired: O5c mixed fusion-only update512, SHA256
  `7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a`.

The backbone and fixed fusion scale are identical across endpoints; only the
two fusion matrices differ. All parameters and buffers must be byte-identical
before and after each completed evaluation case. Load model state only, using
verified checkpoint bytes and source/configuration identities. Do not restore
or construct an optimizer. The existing forward/evaluator math is unchanged.

Use beta1 throughout; finite K2/K3/K4 and exact sequential online feedback.
K counts total passes, including ordinary pass0. Each finite pass reports its
own CE, never the summed training objective. Online uses the freshly completed
previous-token state with fresh per-row caches; it is teacher-forced evaluation,
not free-running generation. Ordinary pass0 supplies the matched backbone
reference. Preserve BF16 mixed, FP32 stored parameters, native SDPA, batch8,
TF32 off, no compile/graphs/distributed, and no prefix mixing or hidden jitter.

## Fixed data grid

Use the unchanged O4 prepared corpus and development splits only, with exactly
the same first windows in every execution and endpoint:

| Section | Windows per domain | Maximum input length | Purpose |
| --- | ---: | ---: | --- |
| short_prefix | 32 | 64 | Comparable to earlier O5b short diagnostic and initial timing |
| full_context | 512 | 512 | Same selection/context as final O5c evaluation |

For each checkpoint run short then full, K2/K3/K4/online within each section.
Source checkpoint first, repaired checkpoint second. Both code (`dev`) and
WikiText (`retention_dev`) use the full-vocabulary, token-weighted CE and
same-document adjacent-target mask already validated in the evaluator. Short
rows remain short; no invented EOS, added targets or document packing.
Different sections are not matched-context scores and must remain separate.
The full extension has 422 original code documents and59 WikiText documents;
the short WikiText selection contains only six original documents.

## Measurements and interpretation

Report final-pass/online NLL, perplexity, accuracy, per-original-document
records, target counts and elapsed evaluation time. Compare:

1. Mixed online versus mixed K2: does the learned repair survive the state
   distribution encountered by sequential feedback?
2. Mixed versus source at each execution: does adaptation help both finite and
   online execution?
3. The endpoint-by-execution interaction: `(online-K2)_mixed-(online-K2)_source`.
4. K3/K4 and the fixed ordinary pass as descriptive references.

Paired intervals use1,000 bootstrap resamples of original documents, seed
20260922, aggregating windows before token weighting. They do not estimate
training-seed variance. No beta selection, learning or test-set access follows
from this diagnostic. Preserve one-seed, development-only and narrow-domain
qualifications; no advantage over an equally additionally trained ordinary
model is established by these checks. NLL agreement does not prove logits or
hidden states are equal, and bounded contexts do not clear arbitrary lengths.

## Operations and recovery

Run only in the project GPU container, checking successful nvidia-smi/CUDA and
matching the completed O5c runtime. Log online to the user's W&B project and
record time from the initial short checks before projecting full cost. Save the
report atomically after each domain/case, with progress every eight batches.
Cases complete only after all state hashes are rechecked. An interrupted run
can reload the unchanged pinned checkpoint and resume after completed cases;
an unfinished case is repeated in full. Do not fabricate optimizer checkpoints
for evaluation-only work: both full model endpoints are already retained.

Retain source/protocol, reports, plots, bootstrap summaries and parent GCS
identities under a new O5d prefix. Preserve failed attempts if any. Stop and
report nonfinite metrics, mutation, identity drift or runtime errors. A poor
quality result is a finding rather than an automatic reason to change the grid.
