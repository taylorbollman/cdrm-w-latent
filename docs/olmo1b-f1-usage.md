# F1 integration and early profiling

F1 implements the first stage of the user-approved
[functionality/execution plan](fbt-rt-nextlat-research-plan-v4.md).
The [protocol](reports/olmo1b-f1/protocol.md) and
[configuration](../configs/olmo_f1_integration.json) define the bounded scope.
This runner performs small nonzero optimizer updates; it is not a quality study.
The initial execution is complete: [results](reports/olmo1b-f1/results.md) and
[assessment](reports/olmo1b-f1/assessment.md). Do not restart it.

## Execute in a new directory

Use the project container; the runner refuses host execution or CPU fallback:

    bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_f1_integrate.py --artifacts .runtime/olmo1b-step60000/artifacts --output-dir .runtime/olmo1b-step60000/f1-integration-NEW'

The output directory must not exist. An optional --cases comma-separated list
selects named cases for a smaller preflight; the report records that subset
and must not be described as a full 18-case pass. --configuration accepts an
explicit JSON matrix under the same schema.

Defaults use all eight RT/FBT/NextLat combinations at B2/T32, four transition/
pass/layer-selection cases, two T128 cases and four B1/T512 early profiles.
RT selects layer0 except the explicit (0,15) stress case. There is no all-layer
RT, new attention kernel, Q/K normalization, compiler, graph or distributed
implementation in this milestone.

All cases rebuild from the pinned original OLMo-1B checkpoint; optimizer states
are fresh per case. The runner uses existing canonical losses and AdamW helper.
It freezes unused registered fusion parameters when FBT is disabled. The shared
NextLat predictor exists only when enabled and is omitted from the reported
inference parameter set. Shared/tied parameters count once.

## Outputs and recovery

The aggregate report.json contains runtime/source/protocol hashes, requested
cases, W&B URL and full completed case records. Each completed case also has
its own JSON file. Progress files retain diagnostic updates before rejection.
configuration.json stores the requested matrix verbatim.

Ordinary and all-three cases save after update2, rebuild/load that boundary and
replay update3. Exact checks cover model, optimizer, scheduler, counters, cursor,
fixture and subsequent CPU/CUDA RNG draws. The synthetic recovery checkpoint
is deleted only after passing. If interrupted or failed, inspect its report
and progress files; do not blindly resume a completed case or overwrite a run.
The runner does not implement resumption of its overall diagnostic queue.

The four T512 profiles use a complete observed update, three warmups, three
timed changed-input/weight updates and one separately profiled update.
Timing excludes fixture construction, gradient-observer hooks, W&B, hashing,
checkpoint writes and the profiler itself. Memory peaks include initialized
optimizer state. Counts distinguish valid data tokens from CE/auxiliary targets
and repeated pass work.

## Tests and reporting

Run focused CPU checks explicitly without GPU passthrough:

    CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc 'python -m pytest -q tests/test_olmo_f1_integration.py tests/test_olmo_f1_observe.py tests/test_olmo_f1_retain.py tests/test_olmo_fbt.py tests/test_olmo_fbt_training.py tests/test_olmo_fbt_adversarial.py tests/test_olmo_lm_training.py tests/test_olmo_tiled.py'

Additional F1 reporting tests live alongside these when available.
Retain the scoped test record in docs/reports/olmo1b-f1/test-results.txt.

After a completed passing run, generate and assess the resource/capability
report:

    CDRM_DOCKER_GPUS=none bash scripts/docker_shell.sh bash -lc 'python scripts/olmo_f1_report.py --report .runtime/olmo1b-step60000/f1-integration-01/report.json --output-dir docs/reports/olmo1b-f1'

Then retain small evidence with scripts/olmo_f1_retain.py. The retainer
requires exact requested-case coverage, matching source/config/protocol records
and per-case JSON, and references the existing verified native checkpoint.
It does not bundle disposable checkpoints. See the completed report and
storage receipt for the final command, artifact prefix and observed limitations.
