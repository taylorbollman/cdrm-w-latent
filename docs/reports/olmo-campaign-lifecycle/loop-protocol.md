# Bounded campaign lifecycle integration

2026-09-29. New opt-in host-loop helper and tiny integration CLI. Existing
pretrained, numerical and checkpoint-recovery evidence remains frozen.

The reusable `scripts/olmo_campaign_loop.py` contains no optimizer math or CUDA
capture. It coordinates host actions at completed optimizer/cursor boundaries.
All ranks execute the same update/save sequence. Rank zero checks a stop-file
and the monotonic checkpoint deadline; every rank may request stopping with a
signal flag. Signal handlers perform no I/O, CUDA or collectives. A shared
boundary decision stops only after the current complete update. Checkpoint
cadence is at most 600 seconds **between boundary checks**; one long update or
I/O cannot be safely interrupted and may exceed that wall-time interval.

A successful optimizer step is followed by committed reader cursor, coordinated
rank-zero metrics, then the next shared boundary decision. Rank-zero tracking,
status and retention failures are propagated to every healthy rank. A logging
failure after a successful update saves that completed boundary and exits as a
failed run. Failed/invalidated model updates or failed in-flight collectives do
not trigger an emergency checkpoint. The launcher must enforce an external
runtime bound and tear down the process group; recovery uses the last committed
checkpoint in fresh processes.

Checkpoint save reuses `CampaignDDPGraphTraining.checkpoint_boundary` and
`save_distributed_checkpoint`: no live-graph in-place load, no new serialization
format. Each checkpoint destination is unique and immutable. Optional GCS
retention creates objects only, checks hashes including a generation-pinned
readback, and uploads the manifest last. The local latest pointer is updated
after successful publication/retention. A retention failure preserves the local
checkpoint and prior latest pointer; it never deletes either.

The first integration CLI is deliberately **tiny-only**: two layers, width32,
heads4, MLP64, native corpus vocabulary/EOS, full FP32/math attention and eager
native RT inside actual captured two-rank DDP. NFR is the default arm, B is also
supported. Four FBT passes, NextLat and the established native recurrence are
exercised without changing their core implementations. This is not a
pretrained model, BF16 agreement test, throughput result, H200 qualification,
new production mixture, or quality training.

Use the existing real tokenized coverage corpus and a separately pinned T16
continuous-stream packed index. Preserve all real token IDs, document IDs,
packing semantics and row-keyed jitter. Physical batch is2 per rank. The three
logical target counts64,128,192 give M1/M2/M3 accumulation on the unchanged graph
shape, with a fixed three-update token schedule. These are diagnostic targets,
not proposed training hyperparameters. No data cycling or production plan
selection occurs. Restore validates rank partition, committed cursor and data
clocks before DDP/capture; preparation must preserve full model/Adam/RNG/clocks.

Acceptance stages, run separately by root after frozen source/CPU review:

1. Uninterrupted three-update reference; immutable checkpoints at0,1,2,3.
2. A new identical run creates a stop-file after update1; all ranks save and
   exit at1 with no second optimizer update.
3. Fresh processes restore the stopped update1 before DDP, rebuild graphs,
   and compare updates2 and3 bitwise with the reference: data/noise, raw
   gradients, metrics, model/Adam/scheduler/RNG and committed cursor.
4. Deliberate rank-zero log failure after update1: a coordinated **failed**
   run, no second update, completed boundary saved. Fresh-process recovery
   from that saved boundary may reuse stage3's exact continuation comparison.

CPU coverage includes fake-clock cadence, stop/signal boundaries, logging
failure saves, failing updates without emergency save, retention publication
ordering, actual packed cursor controls, tiny real model/objective data flow,
and two-process Gloo coordination with an actual optimizer/checkpoint reload.
Root alone runs GPU stages in the project container. Both NCCL async-error
flags are0, process-group timeout180 seconds and external stage limit1200
seconds. Online W&B, persistent reports/source snapshots and verified retention
are required. No stage is claimed successful before its completed report.
