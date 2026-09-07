# Independent CDRM reference review

Reviewed 2026-09-07, with source identities recorded at 13:01:31 UTC. This review
found no remaining blocking defect for the declared single-process FP32 pilot.
It covers the new core composition, targeted tests, and the training/validation
harness. It does not establish a learning advantage or general backend clearance.

## Core and independent tests

The reviewed implementation matches the brief: ordinary blocks 0–8 produce
`p3` and `p8`; queries and temporary K/V come from `p3`; each read uses earlier
permanent records plus the current temporary pair; the current permanent write
is created afterward. The candidate uses the normalized `p8-p3` difference,
and the late bridge uses normalized `hat_m-p3`, with `p8` as its residual anchor.
The active same-depth control substitutes `p3` only in the candidate adapter.
Absolute ALiBi rows and a single attention scale are applied to the restricted
prefix. Memory is local to each call and remains connected to ordinary autograd.

Block 3 owns the shared fused QKV weights, biases, and learned norms. The side
module receives that owner functionally and registers only two bias-free
adapters. Its stateless RMS normalizers do not replace backbone normalization.
The optional state entering block 9 remains hidden-state entry 9; raw `p8` is
exposed separately. The default `cdrm_enabled=False` adds no side parameters.

[The targeted suite](../../../recurrent-transformer/tests/test_cdrm_reference.py)
checks independent FP64 explicit attention/scan references at D16/H4 and D32/H16,
including all side-owner, adapter, and independent-input gradients against
FP32 production at `atol=2e-6`, `rtol=2e-5`. Other checks cover:

- Lambda-zero SEQ logits and all common gradients, tied and untied heads.
- T7 causality, independent examples/calls, and T1 read-before-write behavior.
- Earlier independent deep-leaf credit at rho=1 and its absence at rho=0.
- History-conditioned writes, active same-depth/current-only controls, and the
  legitimately unused terminal write.
- Shared gradients equaling preview plus untied-side gradients, including all
  fused QKV slices and learned norms; unique optimizer identities and storage.
- Paired backbone initialization, nonzero adapters, model/config serialization,
  and bitwise next-update agreement of parameters and Adam state after reload.
- Rejection of unsupported configurations, cached decoding, packing, custom
  masks/biases, autocast, and non-FP32 parameters.

The reviewing agent observed **52 passed, 374 warnings in 1.41 s** in a container
with GPU passthrough explicitly disabled. This is an **agent-reported execution
record**, not a retained raw log; the original output exists only in the tool
transcript. No raw stdout file has been reconstructed or fabricated.

```bash
CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc \
  'test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && python -m pytest recurrent-transformer/tests/test_cdrm_reference.py -q --disable-warnings --maxfail=3'
```

The first bitwise bypass attempt used an attention context entered before model
construction. OLMo's existing constructor subsequently enabled flash SDPA. With
math SDPA selected after construction, all 13 hidden states and logits matched
bitwise. The test was corrected without loosening its tolerances. The current
harness establishes backend flags after construction and scopes forwards with
math SDPA.

## Harness findings and resolution

| Finding | Verified resolution |
|---|---|
| Development epoch 0 was evaluated but excluded from best-checkpoint eligibility. | Its answer CE initializes `best_score`; initial weights can hold the best-development role. |
| A failed `--reference-final` comparison could leave status `complete`. | Numerical differences are retained and a failed comparison raises. |
| SEQ/R3 validation unconditionally requested CDRM diagnostic states. | The request now follows `model.config.cdrm_enabled`. |
| Validation source tracking omitted the validation script itself. | `validation_sources()` includes its own file digest. |
| Trainer shuffle default differed from the retained MAD shuffle specification. | The trainer now uses 45678. |

Native MAD training labels feed the optimization objective without a second LM
shift. Answer labels feed separately named retrieval metrics. For recall, the
native training objective is dense next-token prediction, while evaluation
scores repeated-key values. Configured T128 recall yields actual T127 inputs;
these lengths must remain explicit in reports.

Additional CPU-only helper checks confirmed ignored-position gradients are zero,
CE and token metrics use scored-token counts, and exact match is counted per
example. Epoch permutations reconstruct independently of global NumPy RNG,
including a partial final batch. The cosine scheduler steps after each completed
epoch and retains its 200-epoch horizon: LR after epochs 1, 25, and 200 is
respectively `0.0004999692198041743`, `0.000481007943361566`, and `0.000001`.
Scheduler state restored exactly in the helper check.

The resume path restores model, optimizer, scheduler, Python/NumPy/CPU/CUDA RNG,
completed epoch/update counters, and within-epoch position, then checks state
digests and the next batch. It does not repeat development evaluation or insert
warm-up updates. Its unresolved self-checkpoint role hash is resolved on load.
Pruning protects milestone/latest/best artifacts and only removes files in the
current output directory, preserving parent lineage. Initialization remains a
weights-only artifact, with update-zero resume rejected.

## Precision, topology, and remaining scope

The final read-only pass found no precision or normalization defect requiring a
pre-pilot change. Core guards reject AMP/non-FP32 execution and CUDA TF32; the
harness also disables TF32, uses deterministic math attention, and enforces one
process. Backbone norm modules and their learned parameters are reused directly.
R3 replacement, its BF16 precision policy, GQA, RoPE, dropout, block grouping,
cached decoding, and activation checkpointing cannot be combined with this CDRM
profile. The pilot has no accumulation, distributed wrapping, or custom backward.
Alternative normalization configurations and arbitrary site placements are not
new numerical clearances merely because configuration fields accept them.

The lead reports an additional foundation run of 60 passed/20 skipped and GPU
passes at D256 B2/B64 and D128 recall B2/B128. Those are lead-run evidence, not GPU
executions performed by this reviewer. The CPU optimizer reload test and GPU
one-update round trip do not substitute for the complete runner check. Final
runner recovery evidence was subsequently inspected and is recorded below.
No learning result or broader precision claim is made here.

## Reviewed source identities

SHA-256 values identify the final files inspected for this note, not a claim that
every listed file was frozen at the earlier CPU test start.

| File | SHA-256 |
|---|---|
| `recurrent-transformer/olmo/cdrm.py` | `e47221811489f9cc8757d10521f91814a7c979b5a5ecebf83339b7ce20daf8d9` |
| `recurrent-transformer/olmo/config.py` | `728b9f4e5dd1b82a5d574e12264bc8dae74622d9b285b568642a139a16aa8db6` |
| `recurrent-transformer/olmo/model.py` | `74a9b0819264cd5c2ada9e00373f49d1afc9c64209c2d06a49b9d41833327f17` |
| `recurrent-transformer/tests/test_cdrm_reference.py` | `7a52dc876afb4f9747228d8c62d627e80232146094459d50238db57382e2c41b` |
| `scripts/cdrm_common.py` | `6ed06f2500ac06d1f04207a26f0e1ae209825b51e638137ddba77127f5d191b6` |
| `scripts/cdrm_train.py` | `58fb88cd7b2f896b256d33efdb7980ba60ac7808ce1e7a3859ce1bc1accce1b9` |
| `scripts/cdrm_validate.py` | `d4403a24b7ced733449767cf82f99e72fbab18960929434d0da6b97cea0600ac` |

## Operational write-path addendum

The first real OPS attempt found an integration defect missed by the earlier
read-only review: imported `atomic_json` defaults to `replace=False`, but the
trainer writes its checkpoint-role ledger repeatedly. The attempt completed an
update and saved `epoch-0001.pt`, then raised `FileExistsError` while replacing
the existing ledger. The [failed report](../../../.runtime/cdrm-naive/20260907T123830Z/ops-tiny-fit/report.json)
and its artifacts remain retained. This was an operational failure, not a passed
runner recovery test.

The lead changed only the three owned-directory ledger calls to
`replace=True`. Other JSON outputs remain immutable, learning curves use the
explicit append helper, and model checkpoints keep unique names. The resulting
trainer SHA-256 is
`0741cb7987fe9f6867262c082e008c186ef678fe98a19df1c238acde943a64e6`.
The pre-fix source archive and [manifest](../../../.runtime/cdrm-naive/20260907T123830Z/source/source-num-and-first-ops-v1.json)
are retained; archive SHA-256 is
`226d4fa5249885abfd1e2357d74534a012be1212f358f23a900909862fbceac3`.
The earlier source table therefore identifies the pre-ledger-fix review snapshot.

A subsequent standard-library-only check executed the actual JSON helper and
retention function AST in temporary directories, with a metadata-only checkpoint
save stub. It verified immutable-output refusal, ledger replacement, JSONL
append, milestone/latest/best protection, parent-lineage protection, and removal
of obsolete child checkpoints, including coincident latest/best roles. No model
or GPU ran in that check, and it does not validate tensor serialization.

One additional edge was reported: immediate pause before any new process-local
update could save a checkpoint but then fail the final precision audit because
`require_gradients=True` expected gradients absent in that process. The lead
corrected this to require gradients only after a new update, records
`updates_this_process`, and reran the complete recovery sequence on final v3
source. No computation or optimizer update rule changed in these integration
fixes.

## Final-source GPU operational evidence

The reviewer read the following retained reports, verified their SHA-256 values,
and checked the existence and checksums of their latest/best checkpoint roles.
All four share identity
`2a2deb202c8a13d94984fe4d48e687d22ebcffab3644079dc99e7835c8f76f88`:
CDRM D128/H16/MLP512, physical B32, actual recall T127, 512 fixed training examples
and the next 32 training examples as an OPS monitor. Gates are epsilon=0.1,
lambda=0.01, rho=1. These checks accessed no research development or final labels.

| Report | Direct result | Report SHA-256 |
|---|---|---|
| [Uninterrupted v3](../../../.runtime/cdrm-naive/20260907T123830Z/ops-fresh-v3/report.json) | Complete: 32 updates / 2 epochs. | `3c119375156c6ec2b6b780b1ed8d8c37cc8aae4031f06e1f99e22f8552bed4c1` |
| [Midpoint resume v3](../../../.runtime/cdrm-naive/20260907T123830Z/ops-resume-v3/report.json) | Update 16 → 32; exact comparison passes, no differences. | `4732acfe4fc51b0059e724436918a9bdda3fe6d7e936c801809a07ae01faeaec` |
| [Immediate pause v3](../../../.runtime/cdrm-naive/20260907T123830Z/ops-immediate-pause-v3/report.json) | Status `paused`; retained update 16, zero new updates. | `0da158234f62b2fd35a260ea64b463021ad040f95081b71d53d18d46e622e7f1` |
| [Resume paused checkpoint v3](../../../.runtime/cdrm-naive/20260907T123830Z/ops-pause-resume-v3/report.json) | Update 16 → 32; exact comparison to uninterrupted run passes, no differences. | `96f1658d6811e189a515ede40f606f5dd6ae25afca2a65c6835e8eb3558ea5b4` |

The audited comparison covers model, optimizer, scheduler, RNG, completed counters,
within-epoch position, next-batch hash, identity, non-timing training history,
and development metrics. It excludes wall times, paths, serialized file bytes,
compiler history, and checkpoint-role ledgers. Both recovery reports show exact
restore with no repeated evaluation or warm-up updates. Runs that performed new
updates report 101 finite FP32 parameter/gradient tensors and 202 finite FP32
moment tensors. The immediate pause correctly reports zero gradient tensors
while retaining finite parameters and moments; absent gradients there are not
claimed as a gradient check.

The pause artifact has SHA-256
`662b36acde2426102199426df2938b8627d6f43d35c34fa00c02ad2bcce4972f`.
The final training source hash in all four reports and the current file is
`97eb0a828e22b120bb0f85e14b5b5c75bfe559a6059ae0f2a1da104d5ffcc603`.
The [frozen pilot source manifest](../../../.runtime/cdrm-naive/20260907T123830Z/source/source-pilot-v3.json)
has SHA-256 `0e7d12d11d6988a05b8916869107e12fd6c1afae5d8a8d36dcceb8536173051e`;
its archive has SHA-256
`3da35594e2d8bade58fb26a8152882d93d89e55d6eb0d8266a0d8c59746b7b59`.

This establishes actual GPU recovery for the tested epoch-boundary continuation
and immediate-pause path. Arbitrary within-epoch interruption, B128 bitwise
recovery, other tasks/precisions, and distributed or accumulated training are
not additional claims of this check. The lead has frozen shared recall-50 and
selective-copying-25 epoch pilot endpoints before development inspection; their
pairing and learning results are separate, pending evidence.

## Six-block steering and output compatibility review

Later user steering superseded the earlier paired 12-block plan with SEQ-first
six-block screening. Both candidate site pairs, early/late 1/3 and 1/4, were
added to the reference suite. The original 52 cases and eight added topology,
bypass, causality, canonical-owner, and independent deep-credit cases passed:
**60 passed, 422 warnings in 1.56 s**, CPU-only Docker. The [raw log](../../../.runtime/cdrm-naive/20260907T123830Z/execution/cdrm-six-block-tests-cpu.log)
has SHA-256 `7142e43520062abd317d3f9317dcdd710a4b719764f3f5090927821cb1e457c4`.
This later suite does not retroactively supply a raw log for the original run.

The selected six-block configuration uses early 1 / late 3 and suffix 4–5.
Current owner-gradient diagnostics filter the configured early block; validation
checks hidden-state entry `late+1`. Resolved six-block CDRM configurations have
no recurrent preview replacement. The usage guide and joint manifest explicitly
define `p3`, `p8`, and `v8` as compatibility names for the configured early, late,
and bridged states. These names do not imply physical sites 3/8/9 in a six-block
model. Screening now rejects explicit final-set evaluation in its parser.

A bounded review of current repository output consumers found one real default-
disabled regression: the HF wrapper's `return_dict=False` branch concatenated
`outputs[1:]`, so adding native `cdrm_states` extended the wrapper tuple even when
that field was `None`. Native training/generation and named HF output paths read
named fields and were unaffected. The authorized wrapper-only fix now constructs
the four legacy fields explicitly: logits, attention K/V, hidden states, and
pre-logits. Labels still prepend loss. Native `cdrm_states` remains unchanged.

[Focused wrapper tests](../../../recurrent-transformer/tests/hf_olmo/test_output_tuple_compatibility.py)
run a real CPU native model with CDRM disabled, with/without labels and
with/without an injected output-only diagnostic field. They verify tuple arity,
field identity, cache/hidden-state placement, and loss alignment: **4 passed,
24 warnings in 1.98 s**. The diagnostic injection tests wrapper formatting and
does not claim HF execution support for CDRM. The [raw log](../../../.runtime/cdrm-naive/20260907T123830Z/execution/hf-output-tuple-cpu.log)
has SHA-256 `9c1ced7ce431c73a4c36415c96c3d1547e376884621e9e13ee2559c94a0f5781`.
Wrapper SHA-256 is
`b3aeaa9816faf71f6719692bb819c4aaac1db8cb977a879e32a781d070e37217`;
test SHA-256 is
`d617aa502ec6b324bad7dd16dc834b79ffc903c611fde0c7a229d044edf37a65`.

No further current-repository output-consumer or six-block owner/index regression
was found. No core, configuration, runner, or precision-policy file changed in
this compatibility fix, and the reviewer ran no GPU work. Selection, pairing,
and learning evidence for the harder six-block task remain separate reports.
