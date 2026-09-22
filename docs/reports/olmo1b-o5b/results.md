# O5b: matched FBT recovery pilot — running

The preflight passed and the two-arm learning queue started on 2026-09-22.
**Learning results are pending.** This file will be replaced by the strict final
comparison after both arms complete. [Protocol](protocol.md),
[usage](../../olmo1b-o5b-usage.md), [handoff](../../fbt-rt-nextlat-handoff.md),
and [draft PR9](https://github.com/taylorbollman/cdrm-w-latent/pull/9).

## Scope

Native OLMo-1B step60000 (~252B tokens), with the same code stream/order in both
arms. RT and NextLat are off. Both use two pass losses; the ordinary control
keeps feedback zero and the FBT arm gradually introduces feedback. Budget per
arm: 2,634 updates, 20,855,799 valid input tokens, 20,771,511 CE targets. Effective
batch 32, T512, physical 16 accumulated twice. FP32 parameters/AdamW, BF16 mixed
computation, native SDPA, no compile/CUDA graphs.

Native LR 1e-5 after 100-update warmup. Feedback ramp ends at 1,367; the separate
fusion LR warms to 1e-4 over its first 100 potentially active updates. Prefix
sampling and hidden jitter are off: this differs from the author reproduction
and is a first mechanism/recovery pilot, not a replication or efficacy result.

## Completed validation

The scoped CPU suite passes 693 tests, including 150 new evaluation, runner,
report and retention tests. Source implementation commit:
`4751d508b26887b8873cf43b06acc2b47994067c`.

Actual H100 complete beta1 steps used full-length real train windows, two warmups
and three timed updates per candidate, with initialized AdamW state and LR0:

| Physical batch / accumulation | Peak allocated | Median step | Valid input tokens/s |
| --- | ---: | ---: | ---: |
|32 /1 |74.53GiB |0.909s |18,018 |
|16 /2, selected |49.60GiB |0.919s |17,821 |

Both preserve all model/buffer bytes and have finite optimizer state. The
selection keeps effective batch32 and substantial memory headroom at little
measured timing cost. These are three-sample eager timings, not tuned throughput.

Initial 512-window code/retention NLL is 1.787902/3.040993, matching O4's original
checkpoint. Cold full-strength feedback is highly disruptive: on the 128-window
subset, pass 1 NLL is 9.569364/11.091882 versus same-window pass 0 values
1.690766/3.105747. The gradual ramp tests whether learned feedback recovers;
this cold result neither demonstrates learned benefit nor rules it out.

On identical first 32 short windows capped at 64, beta 0 parallel and exact-online
NLL differ by about 0.0006 nats with BF16 cached/full execution. Report finite and
online measures separately and never compare short-prefix NLL with 512-window
NLL. No new numerical-fidelity claim is implied by finite losses.

## Running and retained evidence

- [Ordinary control](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ichekj67)
  is first; the FBT arm starts automatically after it completes.
- [Preflight](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/upirj0yb)
  is completed. [Raw summary](preflight-summary.json), [CPU tests](test-results.txt).
- Queue: `.runtime/olmo1b-step60000/o5b-pilot-01/queue.json`; automatic CPU
  report/retention finisher writes `finish-status.json` beside it. Read those
  files for current status; this launch note is not a live dashboard.
- Config SHA256:`b26d4f7af7e6888bf0f8720724d2aadc3f4c1ed842dd988ebbedc89545bce626`.
- Verified [initial storage receipt](initial-storage-receipt.json) references
  original native weights and the unchanged prepared O4 corpus without uploading
  duplicate model/data bytes. New prefix:
  `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5b-code-pilot/20260922T040000Z/`.
- Each full optimizer checkpoint is retained and verified before removing its
  older local predecessor. The queue halts on errors or the declared health gate.

Review final pass 0/pass 1 losses, retention, paired document intervals and exact
online behavior before extending exposure or adding another mechanism. This
single-seed ~21M-token recovery check cannot establish architectural efficacy.
