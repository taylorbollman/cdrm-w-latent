# Six-layer embedding experiment series — active authorization

> **Latest user budget change:** positive10k pilots now continue to **25k**, not50k.
> The strict constant-value5k E36>50% condition for running the linear arm stays
> unchanged (it already passed:71.56445%). The linear schedule still reaches
> lambda0.01 at20k. The active constant191000 run is not being restarted: watcher
> **PID20025**, in new192000 queue directory, waits for its atomic025000 checkpoint
> then writes the existing graceful-stop sentinel. A final update may complete;
> the exact25k comparison checkpoint and actual terminal update are kept distinct.
> No model, driver or current protocol file changed. The old190800 queue has
> STOP_QUEUE to prevent additional50k launches, while its active trainer remains.
>
> Replacement queue directory:
> `.runtime/rt-a5/20260915T192000Z-six-layer-embedding-series-25k/`.
> Replacement future pilots:
> `.runtime/rt-a5/20260915T192500Z-six-layer-value-linear10k/` and
> `.runtime/rt-a5/20260915T193000Z-six-layer-head10k/`.
> Their positive decisions create matching `linear25k` and `head25k` lineages,
> with exact15k/20k/25k saves. Old prepared175000linear/180000head stay unlaunched.
> Future helper/report copies are reviewed and both queue-ready receipts exist.
> The new master **PID22561** started19:28:39 UTC and is waiting for the
> active constant25k stop; watcher20025 is armed. Read the new192000 status/launch receipts for
> subsequent updates. Older50k references below describe superseded plans.


> Recovery update,2026-09-15 ~19:10 UTC: the host restarted (current boot19:03:57),
> so all old trainer/queue/retainer PIDs below are historical. The constant-value
> pilot completed10k before the interruption: full5k E36=71.56445% and full10k
> E36=78.64648%. Both gates passed. Its10k checkpoint SHA is
> `20c2f96db145bd9fbd181526d9f498714c13d561e6cc423d2fe8b450449fcf98`
> and was rechecked after restart. The original constant50k attempt has579 intact
> history rows (10001–10579), then2566 NUL bytes, and **no checkpoint**. Preserve
> its raw files; do not strip NULs and do not ingest that unsaved tail as resumed
> training. Recovery resumes the exact10k parent into a new output lineage.
>
> New constant continuation:
> `.runtime/rt-a5/20260915T191000Z-six-layer-value-constant50k/`.
> New queue:
> `.runtime/rt-a5/20260915T190800Z-six-layer-embedding-series-recovery/`.
> New host H100 UUID `GPU-692c1897-cec3-7622-5475-6629b4cbb356`; same
> Torch2.13.0a0+8145d630e8.nv26.06 verified inside project Docker container.
> Recovery queue **PID10987** started19:13:24 UTC and is active.
> Constant continuation launcher10993, retention10990; W&B **rci2vhfb**
> (<https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/rci2vhfb>). The linear and head pilots are still
> unlaunched and remain in the same authorized serial order. The constant10k
> four report figures were reviewed; no findings. Report W&B42ve12e1.


User requested on2026-09-15:

1. Extend current six-layer approach1 fixedlambda0.01 from10k to20k, thenstop.
2. Fresh six-layer approach2 permanent-value addition, fixedlambda0.01.
3. **Only if constant-value E36 is strictly above50% at its full5k evaluation:**
   Fresh six-layer approach2 permanent-value addition withlinear
   `lambda(s)=0.01*min(s/20000,1)` (lambda10k0.005,20k0.01).
4. Fresh six-layer model reassigning one existing head to historical embedding values.

Each of2–4 first trains10k. At its **actual10k saved checkpoint**, use full102400
OOD length36 whole-word exactness E36: if exactlyzero, stopthatarm; ifpositive,
continueits exactstate to50k andstop. Then proceedtonextarm withoutuserconfirmation.
The user explicitly confirmed this linear eligibility correction; otherwise skip
linear and proceed to the head experiment. Constant still follows its10k/50k
rule independently. OneGPUtrainer atatime. Newusersteering overrides thisqueue. Halt on genuinefailure,
not on developmentperformancebelowreference. No finalconfirmation orlatentrollout.
Allarms sixlayers: index0window2 + fivefullRT, NextLat originalobjective. Onlyindex1
gets the requested modification. FullFP32/eager, originaldata/order/B1024/T12.
Fresh arms share all61successfulsixlayerinitialtensors; valuearms addonePe tensor.

## Current input run and 20k continuation

The 10k parent is complete, checked, plotted and archived (289 members):
`.runtime/rt-a5/20260915T171500Z-six-layer-input001-10k/train-injection/`.
[Parent training](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/redf7ybc).
Its full E36 was 29.33594% at 1k, 71.22949% at 5k, and 73.75781% at 10k.
The corresponding uninjected six-layer baseline was 17.03516%, 67.93652%,
and 82.07715%. The parent report is `docs/reports/rt-a5/six-layer-input001-10k/`.

The exact continuation completed and stopped at20k in
`.runtime/rt-a5/20260915T173000Z-six-layer-input001-20k/train-injection/`.
[20k continuation](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/9hrfzx53).
At its full 15k evaluation, E36=82.79102%, token accuracy=97.26647%.
At full20k, E36=76.52637%, token accuracy=95.72572%; baseline20k E36=80.96289%.
The requested20k stop was honored. Source/model/Adam/RNG/global data order were unchanged.
The report is `docs/reports/rt-a5/six-layer-input001-20k/README.md` (report W&B
`dqs2k3rl`). Saved-state, full evaluation, W&B sync, both GCS checkpoints and
root review of all four figures passed. Terminal archive/readback passed:309 members. Archive SHA
`ce934228ce1c2f140d83f92285d490d690df2ca3a6864a3935574fc3ba5d99b7`;
readback SHA`539b64a1069705ef43cccc74182156151dbf3971401486019fbee0c8d6f92e55`.
The archive includes an immutable snapshot of this live handoff.

- Frozen source65: `cb73ced068d1d01557251f2a7f5d43840fb9bfbcc0e05c6f7821e6ce35b7d877`.
- Child protocol: `05b317a79e7e9a145e6c20612647183d957c8eb1e6e3dba9dfec9e2ae9dd12c1`.
- Parent 10k checkpoint: `bad220a59fbd70cfe020eebca3b78996c89032576299449e98515a1ebcbf6a85`.
- Launcher PID 157575; retention PID 158073; CPU finalizer PID 161765.
- Finalizer owns saved-state checking and the continuation report. Do not race it.
- Root actual visual review and archive execution receipts are now written; do not alter closed artifacts.

## Value variants

Agentnextlat100k_report owns new `scripts/rt_a5_six_layer_value.py`, `_train.py`,
`configs/rt_a5_six_layer_value/base.json`; constant/linear variants.
Reuse frozen permanent-value tiled/autogradadapter, unchangedtemporaryselfK/V,
contextualpermanentkeys andotherlayers. Originalembeddingattached, learnedPe,
independentseed1236. Newfresh6layer runs, notresumesofold4layervalue1k.
Userexplicitlyconfirmed thiscorrection. Implementationplan is in
`.runtime/rt-a5/20260915T172920Z-six-layer-value-implementation/plan.md`.

## Embedding-access head design

Rootimplements separate newadapter/factory/driver/config. In blockindex1, reassign
the **last existinghead** (index7 atD512/H8). ItsqueryremainsW_Q norm(u_j),
itspermanentkey W_K norm(h_t), ALiBi/headbiasunchanged. Replace only selected
headhistoricalvalue by `V_E e_t`, usingtheexistingselectedVprojectionrows asV_E.
Noadditionalprojectionparameter orlambda; overall61tensors/19,998,208params,
attentionwidth andpersistentKVcachewidthunchanged. ExistingtemporaryselfK/V path
remainsunchanged (theuserrequestedhistoricalcontribution). Othersevenheadsretain
ordinaryRTpermanentvalues. Allfiveotherlayersunchanged.

Computeattachedrawembeddingvalue outsidecustomtiledfunction, returnitsexplicit
bypassgradient so both tokenembedding and reusedVrows receivegradient. Saved
bypass onlyoneheadwide. Reuse frozenvendoradjoint throughcalllocalwriteproxy,
replacingselectedVslice ratherthanadding. CPUordinaryautogradreference plus one
boundedtiled/naive actualGPUcomparison needed for newwrite rule; no broadsweep.
Testscheckexact61initialvalues, unchangedKVcache/headshapes, selectedsliceonly,
causality, finite shared/reusedV/embeddinggradients andexactresume. Keep allfrozen
historicalsourcefilesunchanged.

## Persistence and closure

Logeverygraphablerunonline taylorbollman/rt-a5-state-tracking; retaincheckpointsto
`gs://fast-chunks/cdrm-w-latent/rt-a5/<lineage>/`.
Create separatefresh andcontinuationoutputlineages, bindactualparent10k/checkpoint,
keepglobalschedule/dataorder onresume. Buildpersistentsequentialcoordinator once
implementations/preflights are ready; do notreturnwithunlaunchedauthorizedqueue.
Archiveclosedsegmentswithsnapshotsoflivehandoff/status. Neverrewritehistorical
frozenprotocols merelytoextend.

Previousfourlayercontrolclosed:5811observed/5000saved,full5kE36zero,217archive
members. Fourlayerlinear neverlaunched,nowdeferred. Conservativequadratic9665closed.

## Prepared implementations and queue

Value implementation is frozen and reviewed: source65
`1564fc5c1d152fc1132e55083d18839a4ba175866a6c87a825d939cdea86e30e`.
Thirteen CPU model/resume checks passed. Pilot helper directories:

- `.runtime/rt-a5/20260915T174500Z-six-layer-value-constant10k/`
- `.runtime/rt-a5/20260915T175000Z-six-layer-value-linear10k/`

Both use `train-value/`. Each has preflight, launch, saved-state checker and
conditional extension preparation helpers. The generic value reporter passed
five CPU checks and independent review.

Head implementation is frozen and reviewed: source66
`af1fae94e25dfc6982b549157d47ed9492e0e2896a368741a5ec6c39274a434f`.
Five CPU model/resume checks passed. Its helper directory is
`.runtime/rt-a5/20260915T180000Z-six-layer-head10k/`, using `train-head/`.
The head reporter passed five CPU checks and independent review. Both new
families still require their bounded GPU preflights before training; these run
serially through the container after input20k ends.

The series coordinator is
`.runtime/rt-a5/20260915T174000Z-six-layer-embedding-series/run_series.py`.
Its `request.json` records the latest confirmed thresholds. `status.json`,
`active-command.json`, `completed-stages.json` and `linear-eligibility.json`
record actual progress. Each reviewed pilot requires `queue-ready.json`.
The coordinator waits for input20k and its existing finalizer, then executes
preflight, training, saved-state checking, W&B reporting, checkpoint retention,
and the actual saved10k continuation decision in serial order. It never
restarts an existing completed stage. GPU work goes through the project Docker
launcher; report/state tensor work uses the same container with GPUs disabled.

The generic `retain_stage.py` passed eight host checks and independent review.
It uploads each saved checkpoint to the stage-specific GCS prefix and checks
remote checksum/generation at completion. `retention-startup.json` binds the
watcher's live PID, protocol and source before training begins. Training/report
completion is distinct from final visual review and evidence archive closure.

To change the queue, first write `STOP_QUEUE` in the series directory. This
prevents another stage from launching. To stop the active training stage at a
resumable boundary, write `STOP_AFTER_UPDATE` in that stage's lineage directory.
Do not kill the trainer unless graceful stopping fails. A failed or stopped
stage halts subsequent stages for inspection. Historical checkpoints, protocols,
raw logs and source snapshots must remain unchanged.

The reviewed coordinator was launched at 2026-09-15 18:05:56 UTC, PID **175400**.
It is waiting for the active input20k stage and its finalizer. Its source SHA is
`c9db49efe466cc15f3a1eb6e310dd2990f3392ded32c71eb088331949212efab`;
execution-review SHA is `4d62c3be06468f0494a4ff8c9727eaed8ad954e4e40f1d61d4cccbdf2a64d064`.
All three pilot queue-readiness receipts are present. No second coordinator
should be started; inspect the persistent status and live PID first.

## Active constant-value pilot

Started2026-09-15 18:10:17 UTC after input20k ended and its state/report finalized.
[Training W&B](https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/snqwvd5w).
The reviewed source65 remains `1564fc5c1d152fc1132e55083d18839a4ba175866a6c87a825d939cdea86e30e`.
Actual protocol SHA:
`6abd98f3edc68b627747132a73490907bab35aec1b3651820f510d14da97be22`.
GPU launcher PID177945; retention watcher177943; master coordinator175400.
Its bounded GPU preflight passed: all61 baseline initial tensors exactly paired,
FP32 tiled/naive full-model B2/T6 gradients (largest absolute error3.28e-7,
largest tensor relative-L2 error1.06e-6), three discarded actual B1024/T12 updates,
and T36 causality. Preflight W&B8gyfq3nw. Training began from untouched fresh
initialization. Initial checkpoint0 is already retained. Full5k and10k results
remain pending; do not apply the earlier input-injection metrics to these gates.

The head/linear arms remain queued and untrained. The master handles actual
pilot10k decisions and exact positive50k continuations. At the end it reports
`training_and_reports_complete`; final visual review and evidence archives
for those later stages remain a distinct follow-up. All checkpoint retention,
saved-state checks and online reports are automatic within the queue.

Constant-value full1k E36=29.67090%, token accuracy=83.07536% on102400 words.
Checkpoints0 and1k are saved and retention is watching; the5k eligibility and10k
continuation gates remain unevaluated. The original input20k stage is fully
closed, while master175400 continues the authorized constant/conditional-linear/head queue.


## Recovery launch and verification

The fresh queue190800 started2026-09-15 19:13:24 UTC, PID10987. Source SHA
`8f38ef54e3ee2d712472f5b73f6af3f51fe076106fc3543f1224b43c4c369d0e`;
execution review SHA`3e248fee1f4b839f721441afbd70ddd02168f138975e8706c45e81d3295ff916`.
Only the constant50k extension pathname changed from the earlier reviewed queue.
Completed input20k and constant10k were reused without training/evaluation reruns.

Recovery constant protocol SHA
`3c0b15f81c8fe33e0347ce8e332bf47afd68a4d0e77e57ef57897511d69dd2c4`.
The original65 training sources, initialization, checkpoint, optimizer, RNG and
order contract remain fixed. Only output directory, stop path, W&B group/name,
report directory and GCS destination changed; an additive recovery record links
the abandoned attempt. New report output:
`docs/reports/rt-a5/six-layer-value-constant50k-recovery/`.

`recovery-observation.json` compared the first50 already recorded resumed updates
(10001–10050) with the intact old log prefix: every data-order hash, counter,
loss component, accuracy and gradient norm matched exactly. No extra model run
was performed. The old579-row tail is preserved but excluded from stitched
recovery reports. At19:14:21 the new run had reached10259 with finite losses.

Current stop controls are the **new190800 series STOP_QUEUE** and **new191000
stage STOP_AFTER_UPDATE**, not the old dead-process directories. New user
steering overrides the queue as before. The interrupted raw old status fields
are deliberately not rewritten; see old174500 constant50k `interruption.json`
for actual disposition and its separate interruption-evidence archive.


## Active25k queue,2026-09-15 19:28 UTC

New192000 master PID22561 is running and waiting for constant25k. Its source SHA
`ecd00b46015214b7fee2c60e2a79a6b69196d08c7406f3e484e7729d138c51ed`;
execution-review SHA`7af3078b5b1c33d9f0ffa00cbd6403accd43205e4dfb07d59fdad47ffb2bccea`.
Stop watcher20025 has live startup/heartbeat and targets exact025000 atomic save.
Original190800 master10987 still waits for its launched trainer, but itsSTOP_QUEUE
prevents any follow-up; its later halted status is expected after this authorized
budget change. New192000 owns the subsequent CPU finalization and future stages.

The active constant remains in recovery191000, W&Brci2vhfb, launcher10993,
retainer10990. Its full15k E36 is84.13281%. After its planned graceful stop,
`constant25k-completion.json` binds the exact25k checkpoint/full metric and records
any actual terminal extra update separately. Existing50k-family checker/report
and retainer support that explicit stop without changing their frozen sources.

New linear192500/head193000 pilot helpers and25k reporters are independently
reviewed; seven focused reporter checks and seven master mocks passed. New
retainer/stop helper14 host guards passed. All65/66 model sources are unchanged.
Exact positive continuations use checkpoints15k,20k,25k, and the new decision
stringcontinue_to25000. Both prospective queue-ready receipts are bound by the
new master execution review. Current/future graphs remain online; checkpoints
remain retained to each lineage's GCS prefix. No final confirmation or latent
rollout is introduced.
