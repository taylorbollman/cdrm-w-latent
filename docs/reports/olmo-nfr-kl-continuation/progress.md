# NFR KL continuation progress

2026-09-30, approximately 10:33 UTC: activated the predeclared
[protocol](protocol.md) after the completed FBT-only study and bounded
online/component checks. The user authorized evidence-driven continuation
without another review pause during the overnight window.

The F-only segment completed 128 finite updates and retained first-pass CE
(2.687038, versus ordinary control 2.687618), while fourth-pass CE improved
to 2.942000. Its independent native and full B-prefix audits passed. All F,
NF and NFR saved-state curves settle by K32 on the same eight-row panel.
NF32 and NFR32 settle faster than F32 despite much worse CE. The previous NF
KL0.1 continuation improves settled CE while still settling by K32. These
observations support testing loss balance in NFR rather than changing its
fusion gate or pass schedule to address an unobserved failure to settle.

Both branches resume the **original NFR32 checkpoint and populated Adam**.
Neither starts from the new F128 model. Control uses KL1; reduced uses KL0.1.
All other settings, including latent weight1, RT0/15, K4, T1024, physical B12
per GPU, logical 524,288 inputs/update, data order and LR schedule, remain fixed.
Each adds 32 updates and stops at64. The immutable scope and 215 training-source
pins are unchanged. F64 is descriptive same-panel context, not a matched fork.

Activation authority:
`.runtime/olmo-nfr-kl-continuation/activation-01.json`, SHA256
`c9b209280ac92c7b1a86655e90a52bfffe6a47f155c3e9cfb86586a7eb75d2e3`.
It pins F128, its audit, all five completed diagnostics, the declared NFR scope
and the host queue. The online diagnostic's helper completed and synchronized;
its host launcher rejected the spelling `complete` versus `completed`. A
separate adoption receipt preserves that reporting-only mismatch without
rerunning or rewriting the evaluation.

Host queue: `.runtime/olmo-nfr-kl-continuation/sequential_queue.py`
with `--queue-name queue-after-f128-01 --activate`. It runs control then reduced,
requires verified/synchronized control64 before starting the second branch,
and stops on failure or the shared STOP file. No automatic retry or additional
training beyond the paired endpoints. Each branch retains the accepted
asynchronous cloud checkpoint policy and a 10,800-second timeout.

Expected duration is about 3–4 hours. Beginning around 10:33 leaves roughly
three hours to the initial 13:37 target; completing the matched pair may extend
modestly beyond that target under the user's earlier timing flexibility. This
is a finite follow-up, not an open-ended training extension.

Read-only status: `python3 .runtime/olmo-nfr-kl-continuation/live_status_v2.py`.
Reports, launch logs and queue state are under that runtime directory; large
states go to `/mnt/localssd/cdrm-checkpoints/nfr-kl-continuation/` and verified
cloud copies to the declared `gs://fast-chunks` namespace. After interruption,
inspect those authorities before launching anything.

10:34 UTC: queue activated, hostsession57288; readonly monitor96042.
Control started successfully in verified two-H100 container,
W&B https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/ujz924fj .
Both GPUs reserved for this queue; all post-F GPU diagnostics already exited.
Reduced remains queued, not yet launched. Startup/graph preparation underway.

10:38 UTC: Root readonly monitor restarted as53534 after its display code
encountered the valid startup evaluation placeholder with empty panels.
It now waits for completed panel results. Training and queue were unaffected;
old monitor96042 exited. Startup evaluator is repeating saved32 before updates.

10:51 UTC: graph preparation and the first resumed control update are complete.
Update33 is finite, preclip norm8.8348; clipping remains active. The repeated
update32 development CE is3.043012/7.026269/7.055292/7.063815, matching the
parent panel. The next scheduled development measurement is48. Both GPUs
remain reserved; no new diagnostic is being run beside the pair.

11:36 UTC: control48 development completed. Pass1–4 CE is
2.983241/6.764267/6.839343/6.858271, versus restored32
3.043012/7.026269/7.055292/7.063815. All first16 resumed updates are
finite/clipped. Improvement is modest and feedback remains much worse than
first pass. Raw latent losses are.261257/.043548/.044086/.043945; raw KL
2.508881/.620044/.638946/.637515. Later-pass KL rises while CE and latent
loss fall. No objective-total comparison or coefficient conclusion is made
before the reduced branch. Checkpoint43 is verified remotely; the named48
save follows evaluation. Control continues unchanged to64.
