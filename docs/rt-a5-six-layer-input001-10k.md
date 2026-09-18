> **Updated authorization:** continue this model to20k, then run the
> [six-layer embedding series](rt-a5-six-layer-embedding-series.md).

# Six-layer L1R RT + NextLat with constant input injection

## Latest user request

Return to exactly the successful six-layer implementation: first block(index0)
window2 and five full RT blocks(indices1–5), NextLat unchanged. Add approach1 only
to index1 input: `x_t^(2) = u_t + 0.01 P_e e_t`, then its existing normalization.
Lambda is constant0.01 from the start, not learned, with no ramp. P_e is learned,
bias-free,512×512, seeded1236, same projection recipe as previousinputdiagnostics.
The other RT blocks receive no injection. Initial budget10k; user may stop early.
No automatic extension or linear/value experiment is queued.

Use original six-layer D512/H8/GELU2048/ALiBi/Mitchellactualdepth6,
B1024/T12,FP32,originalNextLatCE+SmoothL1(detachedtargetonly),originalAdamW,
fresh seeds1234/1235 and identical800kdata/order. Confirm shared original tensors
exactly equal the successful six-layer checkpoint0 before launch. No frozen
historical training files may change. New adapter/driver/config only.

Reference `.runtime/rt-a5/20260915T141414Z-l1r-six-layer-nextlat10k/`;
W&Bfg4quqku; full10kE36=82.0771%,M36=97.0899%.
The20kcontinuation is separate and closed.

## New experiment

Lineage `.runtime/rt-a5/20260915T171500Z-six-layer-input001-10k/`.
Training `train-injection/` is active, launched17:15UTC by coordinatorPID139548.
W&B https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/redf7ybc .
Protocol SHA256 `a37e476bf3f37266cce49803d0a849153dee3e80ee03b1d0a63b28c4d1807f64`.
Source65 SHA256 `cb73ced068d1d01557251f2a7f5d43840fb9bfbcc0e05c6f7821e6ce35b7d877`.
New files `scripts/rt_a5_six_layer_input.py`, `_train.py`, and
`configs/rt_a5_six_layer_input/base.json`; original62 sources unchanged.
20,260,352 parameters in62 tensors, of which61 are bitwise identical to the
successful baseline checkpoint0. Three discardedactualB1024/T12 FP32 updates
and T36causality passed;10CPUtests passed includingλ0 identity and exactresume.
Preflight W&B `dly4rr1n`; warmed step times about0.150s.
Read liveprogress with `python3 .runtime/rt-a5/20260915T171500Z-six-layer-input001-10k/status.py`.
For an explicit new userstop, write `STOP_AFTER_UPDATE` in thislineage; newdriver
will save the exactcompletedupdate and full terminalevaluations gracefully.
Do not create a stopfile or duplicate launch without usersteering.
Check current statuses before starting GPUwork. All GPUcommands must run in
projectDockercontainer; fullFP32 and no broader numericalcampaign.
Graph training/eval online under taylorbollman/rt-a5-state-tracking.
Retaincheckpointsto gs://fast-chunks/cdrm-w-latent/rt-a5/<lineage>/.

## Superseded control/linear work

Four-layer control `.runtime/rt-a5/20260915T164000Z-l1r-four-layer-control10k/`
was stopped viaSIGINT on usersteering. Its existingdriver does not implement
graceful current-update checkpoints, so latest resumable state is5k; later observed
updates are recorded separately in `user-stop.json`. Do not relabel raw failed
launcher/KeyboardInterrupt as numericalfailure or claim a10kresult.
Full5kE36=0/102400, M36=35.1854%, L12wholeword=98.1934%.
Waitingfinalizer130380 terminated; retentionagent closinginterruptedlineage.
W&B https://wandb.ai/taylorbollman/rt-a5-state-tracking/runs/cu7yhsc2 .

Prepared linear `.runtime/rt-a5/20260915T170000Z-input-linear10k/` never launched.
Its source/tests/review are retained as deferredwork. Do not start its oldgate,
preflight or trainer automatically. The conservativequadratic9,665stop is closed
with286archive members; do not modify those archivedmembers.

Root-bound `implementation-review.json` records actualGPUpreflight/protocol and
CPUtestreceipt. Runtimechecker/report/retention are being prepared independently
while training runs. At terminalstatus, run CPUcheck_final_state then pairedreport
against the original6layer baseline at commonretainedsteps; reviewfigures and
retainterminal evidence. No broaderprecisiontests or additionalexperiments.

## Early result at1k

Full102400-word evaluation: inputE36=29.3359375%, M36=83.23787435%;
L12wholeword=88.98046875%. Original6layer at1kE36=17.03515625%,
M36=75.62673611%. This shows the constant0.01 input injection does not prevent
early36-token generalization in this seeded6layer setup. It is not a multiseed
claim of improved performance. Continue to10k unless usersteers.

Retentionwatcher PID142922 is active. Source/runtime reviewer found noissue;
checker `check_final_state.py` SHA256
`298cfa9865d5178391128924dcca4a7482b64453452530c2d227bfb4a957ee14`
supports a full10k endpoint or explicit earliergracefulstop.

## Terminal workflow

CPUfinalizer PID144951 (`finalize.py`) waits for trainercomplete/stopped, runs
`check_final_state.py`, then `python -m scripts.rt_a5_six_layer_input_report`.
Its `finalization-status.json` ends `ready_for_review`; root still needs to inspect
actual figures, summarize results, and authorize/bind terminalarchive/readback.
It never starts anotherGPUrun. Independentreportreviewpassed7boundedchecks.
Reporter SHA256 `c392851cf4a5ce35481da15116f2bdc08c26d2498dc3b1d8adbd97958be58ea0`.
Output `docs/reports/rt-a5/six-layer-input001-10k/`.

The interruptedfourlayercontrol is nowclosed:217verifiedarchivemembers, archive
SHA256 `7843eaf8d0a015790872e86e777a02a3e79a5d2eaca90f59e170f57e7a4b7965`.
Preserves5811observed/5000saved, no10kclaim. Do not editarchivedcontrolmembers.
