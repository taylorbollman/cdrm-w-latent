# Ordinary OLMo fused RoPE and AdamW: bounded numerical assessment

2026-09-24. Runtime `b70b3ec54cd808534f25e863c61c5cb987c19708`.
Four actual-checkpoint correctness reports completed. Three pass their entire
screen; Dao RoPE at T2048 retains a loss-only numerical failure. All four pass
their own graph and complete-update checks. This is a functionality and bounded
implementation comparison, with no long quality training.

## Reference and scope

The reference is the native original OLMo-1B step60000 checkpoint, with PR26's
rounded compiled ordinary SwiGLU, PyTorch Flash SDPA, all-layer activation
checkpointing, reused native RoPE tables and full CE chunks of 2048. Execution
uses BF16 autocast with FP32 parameters, residuals, RoPE arithmetic and Adam state;
TF32 and autocast weight caching are disabled. All 16 layers are ordinary.
RT, FBT and NextLat objectives are absent. The frozen unused fusion parameters
are not part of the 65 active gradient tensors checked here.

The Dao candidate uses the installed out-of-place split-half rotary kernel,
with FP32 Q/K inputs and FP32 native-derived tables, then restores the projection
dtype. The optimizer candidate changes only scalar AdamW to fused AdamW. In this
document, **combined means Dao RoPE plus fused AdamW**, not the RT/FBT/NextLat
combination elsewhere in this project. These cases do not clear those earlier
RT precision qualifications.

## Same-state comparison

Every comparison begins at the same checkpoint, tokens, labels and weights,
before clipping or an optimizer update. All raw-gradient ownership and finite
checks pass. Values below are ratios, not percentages; maximum error is divided
by the reference tensor's peak magnitude.

| Candidate and shape | Global gradient L2 | Worst tensor gradient L2 | Worst tensor gradient maximum | Output L2 / maximum | CE relative difference | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dao RoPE, B8/T512 | 0.00695926 | 0.00808382 | 0.0126263 | 0.00570911 / 0.0185101 | 8.03430e-6 | Pass |
| Fused AdamW, B8/T512 | 0 | 0 | 0 | 0 / 0 | 0 | Pass, bitwise |
| Dao + fused AdamW, B8/T512 | 0.00695926 | 0.00808382 | 0.0126263 | 0.00570911 / 0.0185101 | 8.03430e-6 | Pass |
| Dao RoPE, B2/T2048 | 0.0113239 | 0.0141340 | 0.0244898 | 0.00645109 / 0.0116529 | **8.53921e-5** | **Fail: CE only** |
| Prespecified limits | 0.015625 | 0.03125 | 0.0625 | 0.015625 / 0.0625 | 1e-5 | Unchanged |

At T512, the largest per-tensor gradient L2 occurs in block 8 `ff_out.weight`;
the largest maximum ratio occurs in block 0 `ff_proj.weight`. At T2048 those
are block 7 `ff_out.weight` and block 9 `ff_proj.weight`, respectively. Block
indices are zero based. Every one of the 65 gradient tensors remains within
both per-tensor budgets at each shape.

The optimizer-only initial comparison is bitwise because no optimizer step has
yet happened. The identical same-state values for Dao alone and Dao plus fused
Adam are expected for the same reason; this does not establish cross-optimizer
trajectory identity.

At T2048, the absolute CE-sum difference is 0.0302429199 over 4094 predicted
positions, or **7.38713e-6 nats per predicted token**. The reference mean CE on
this repeated-text fixture is 0.0865084. A small absolute difference therefore
exceeds the strict relative threshold by 8.54 times. At T512 the corresponding
difference is 3.01593e-6 nats/token over 4088 positions, with reference mean CE
0.375381. These are fixture-specific values, not held-out language-model quality.
The small absolute T2048 difference is useful context, but does not erase the
failed gate or justify claiming numerical equivalence across all contexts.

Native and Dao FP32 rotary operations need not be bitwise: the installed fused
kernel can contract multiply/add operations. The full-stack measurements above
characterize the observed difference; they do not independently localize every
downstream difference to a particular arithmetic instruction.

## Exact checks within each candidate

All four reports pass initial eager-versus-graph backward, changed-token and
repeated-overwrite backward, and changed-weight backward with **bitwise loss and
all 65 raw gradients**. Each also passes three eager versus three graph AdamW
updates, including exact metrics, model/optimizer/scheduler/counter digests and
finite state. The parameters actually change. Fused-Adam candidates record
`fused=True`; scalar candidates record `fused=None`, with `foreach=False` and
`capturable=False` throughout. The optimizer remains outside the captured graph.

Dispatch checks observe actual compiled SwiGLU and Flash SDPA without FA4
fallback. Dao cases record 64 rotary-loader calls during the checkpointed
forward/backward probe; the native-RoPE optimizer-only case records zero.

These four reports contain 24 own-parity optimizer updates, plus six disposable
fixed-gradient optimizer updates described below: **30 physical updates total**.
There are 25 report gates, 24 passed and the T2048 same-state gate failed.
All 21 operational gates pass, including dispatch and the fixed-gradient probe;
three of four same-state numerical gates pass. The failed report is retained
as failed after completing its bounded operational diagnosis.

## Fixed-gradient fused AdamW comparison

One B8/T512 native-model raw gradient was saved and restored before each of
three scalar and three fused AdamW updates. This isolates the optimizer from
subsequent changes in model-produced gradients. Both arms use LR 5e-6, 1e-5, 1e-5,
betas (.9, .95), epsilon 1e-8, native weight decay and max-norm 1 clipping. The
pre-clip norm is 3.37604213 at every step in both arms.

| Quantity | Observed difference | Prespecified limit |
| --- | ---: | ---: |
| All moments, global relative L2 | 2.08772e-7 | 1e-6 |
| Worst moment tensor relative L2 | 2.56814e-7 | 1e-6 |
| Worst moment tensor maximum ratio | 3.34988e-7 | 1e-6 |
| Final weights, global relative L2 | 7.27417e-8 | 1e-6 |
| Worst final-weight tensor relative L2 | 7.31319e-8 | 1e-6 |
| Worst final-weight tensor maximum ratio | 2.37515e-7 | 1e-6 |
| Cumulative update, global relative L2 | **4.56875e-5** | **1e-3** |

The update difference is about 0.00457% of the scalar update's L2 magnitude.
It is checked separately because near-identical pretrained weights could hide
a materially different small optimizer update. The largest per-tensor update
L2 ratio is 0.000221575 at the tied embedding. The largest update maximum ratio
is 0.00869565 in block 1 `ff_out.weight`, with absolute difference 2.38419e-7.
This maximum is descriptive under the prospective protocol: small updates are
quantized at the FP32 spacing of their parent weights. The separate global
update gate and all moment/final-weight tensor maximum gates still apply.

Ownership, all optimizer-group settings except the fused selector, clipping,
LR/scheduler values, counters and step values match exactly. Adam's step tensor
is on CPU in the scalar arm and CUDA in the fused arm, as expected; both end at 3.
FP32 moments are present for all 65 active tensors. Initial weights and raw
gradients are restored in their original storage after the probe. The result
passes every fixed-gradient budget, while remaining a three-update diagnostic
rather than a long-run trajectory or convergence claim.

## Evidence and practical conclusion

Raw reports are under `.runtime/olmo-ordinary-fusions/`, with matching run names:

- [Dao B8/T512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/p4gw0mbw): `correctness-dao-b8-t512-01/report.json`.
- [Fused AdamW B8/T512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/xfsnf66v): `correctness-fused-adam-b8-t512-01/report.json`.
- [Dao plus fused AdamW B8/T512](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/lka71z08): `correctness-combined-b8-t512-01/report.json`.
- [Dao B2/T2048, retained failure](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/haehslis): `correctness-dao-b2-t2048-01/report.json`.

Fused AdamW has the cleaner numerical result in this bounded scope. Dao RoPE
passes the unchanged T512 screen and its own operational checks at both tested
lengths, but T2048 remains qualified. Performance decisions should use the
separate full-step timing results, while retaining this distinction. Production
defaults and historical resume identities are unchanged; none of these checks
substitutes for RT/FBT integration, distributed execution or a quality run.
