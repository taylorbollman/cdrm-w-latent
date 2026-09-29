# Lifecycle helper usage

This is a new opt-in helper; existing campaign and numerical startup runners are
unchanged. Integrate it in a later runner only after defining synchronized safe
boundaries and its last-complete-checkpoint recovery policy.

```python
from scripts.olmo_campaign_lifecycle import coordinated_boundary_action

# Every participating rank executes this call in the same order.
coordinated_boundary_action(
    "log completed update", lambda: tracker.log(metrics), rank_zero_only=True,
)
```

The callback runs only on rank zero within the selected process group. Other
ranks receive `None`; results are not broadcast. Omit `rank_zero_only=True` for
independent local work on every rank. The phase must be a short static label.
Never pass a credential, example, prompt or dynamic exception message as a label.
Validation and phase/ownership agreement occur before callbacks. A callback
exception raises the same `BoundaryActionError` on every participating rank,
including its phase, origin rank and exception class, but no exception message.
The process-group communication backend and timeout remain caller-owned.

Use this only outside graph capture and in-flight model collectives. Do not wrap
a callback that enters a collective on only some ranks. All ranks must reach the
helper in matching order. A failure can leave successful callbacks' side effects
in place; there is no transaction, automatic retry or rollback. A training runner
must stop rather than advance its optimizer or committed data cursor after the
boundary fails. Restart from its last complete published checkpoint, replaying
completed-but-unsaved work as needed. Rank death and hung callbacks still need
external supervision and bounded process-group timeouts.

Run the CPU tests with GPU passthrough explicitly disabled:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python -m pytest -q tests/test_campaign_lifecycle.py'
```

Create a fresh retained namespace for the complete two-rank check:

```bash
CDRM_DOCKER_GPUS=none CDRM_FLASH_ATTENTION_SOURCE=installed \
  bash scripts/docker_shell.sh bash -lc \
  'python scripts/olmo_campaign_lifecycle_check.py --output-dir .runtime/olmo-campaign-lifecycle/check-01'
```

The namespace must not already exist. The harness uses CPU Gloo and tiny actual
Adam state; it does not initialize CUDA or contact W&B. Its static fault checks
are not training curves. Report, per-rank observations, checkpoint fixtures and
four exact source snapshots are stored locally. Tests and a later retained run
do not qualify NCCL/CUDA graph fault recovery or prove that old runners now use
the helper.
