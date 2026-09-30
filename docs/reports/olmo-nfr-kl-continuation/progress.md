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

12:24 UTC: control completed all32 resumed updates and the final64
development evaluation. CE passes1–4 is2.930108/6.473414/6.604760/6.646855.
First/fourth improve.112905/.416960nats from32, but feedback remains
3.717nats worse than first. All32 updates are finite/clipped; raw norm
min/median/max/final is3.52624/7.71443/23.63049/6.13940. Raw KL64 is
2.299452/.666133/.683167/.683961; latent is
.242130/.043243/.042835/.042737. Final checkpoint64 publication and W&B
closeout are pending, with cloud63 verified. Queue retains its terminal
guards before starting reduced from the original32 state. The current
pace places paired closeout around14:30UTC, about an hour after the initial
13:37 target under the user's timing flexibility; no further training
will start after this finite pair.

12:32 UTC: control terminal64/cloud64/W&Bsynced completed with host exit0.
The unchanged queue passed its terminal guards and launched the reduced
branch on both GPUs. W&B:
https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/1xu07xdf .
Original32 model/Adam/data/RNG and schedule are restored; only KL1→.1 changes.
Control terminal evidence retention and summary reconciliation run on CPU
in parallel. No extra GPU diagnostic or training beyond reduced64 is queued.

12:50 UTC: reduced first update33 completed finite. Both ranks' raw
CE/KL/latent loss means match the control's first update exactly. Raw norm
is3.237772 versus control8.834785. The repeated FP32/no-jitter dev32
evaluation also matches all raw component means exactly. This authenticates
the observed shared forward starting point; a smaller gradient norm alone
is not evidence of improved prediction. Reduced48/64 remain pending.
Control terminal outputs and all8 publication receipts are retained, with
W&B terminal checkpoint summary corrected to64/64/not-pending and history
unchanged; see storage-receipt.md.

13:35 UTC: reduced48 development completed. CE passes1–4 is
2.790368/6.348725/6.456982/6.493014, lower than control48
2.983241/6.764267/6.839343/6.858271 on every pass. First/fourth
advantages are.192873/.365257nats. Raw KL is
3.724618/.783862/.822513/.820431; latent is
.341310/.057770/.056030/.055411, higher than control on every pass.
Both branches remain finite; this is a predictive/auxiliary-loss tradeoff,
not improvement in every objective. Reduced continues unchanged to64, with
checkpoint43 verified and named48 publication next. Final paired audit,
summary and review follow terminal64; no further training is queued.

14:24 UTC: reduced64 development completed after all32 finite/clipped updates.
CE is2.774970/5.375003/5.481358/5.520121; first/fourth improvements versus
control64 are.155138/1.126734nats. All raw latent/KL losses remain higher than
control, preserving the predictive/auxiliary tradeoff. Reduced norm
min/median/max/final is2.09518/3.77819/7.92970/4.68710. Feedback still does not
beat the first pass. Cloud63 is verified; final64 publication/host closeout
and then independent paired audit/summary remain pending. No further training.
