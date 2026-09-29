# Background checkpoint retention and pilot preparation

Implementation and tiny recovery acceptance are complete. A four-update native
accumulated-batch diagnostic is running; final native measurements will replace
this progress note before closeout. No 32-update learning cohort has started.

After the normal immutable SSD snapshot and its preservation checks, training
can proceed while one background worker validates local bytes, uploads them,
performs full generation-pinned readback verification, publishes the recovery
receipt and applies the existing keep-two local retention. There is no unlimited
queue. The loop drains before another local save and at normal termination.

Cloud work runs in a fresh CPU-only child process. The training process never
wraps a concurrent upload in process-wide RNG save/restore; that would risk
rewinding live training randomness. Only the main thread updates W&B, reports
and collectives. Historical model math, optimizer, graph, data, saver and storage
verification code remain unchanged. New versioned host orchestration is explicit
in checkpoint identity, so transport changes cannot silently become same-lineage
resumes.

If the VM disappears mid-upload, recovery uses the previous verified cloud
checkpoint. A complete local SSD save is not yet cloud durability. Cloud
publication can finish before the trainer's next boundary poll records it;
persistent publication receipts and the journal remain authoritative. The first
fresh checkpoint can also overlap setup, so a new run has no new durable origin
until that first publication succeeds. Normal final termination waits for the
last checkpoint to finish.

The 480-second timeout bounds the cloud child phase. It is not a hard deadline
for local hashing, fsync, in-progress publication or a complete optimizer update.
These distinctions apply on both spot and reserved machines. A reserved machine
can justify a longer future cadence, but this milestone keeps the accepted
600-second cadence while measuring overlap.

The [test ledger](test-ledger.md) records 136 passing CPU tests and exact tiny
two-GPU blocking/async plus cloud-resume acceptance. Native results remain
pending. The [pilot plan](pilot-plan.md) now has concrete CPU-resolved B and
paired NF/NFR declarations at T1024, 524,288 inputs/update, an initial stop at32
within a declared128-update ceiling and a fixed65,536-input development prefix.

Heavy clipping and worse later-pass CE remain an explicit
[open adaptation issue](open-issues.md), with review criteria at the first pilot
stop. Functional checkpoint acceptance is not useful refinement or a numerical
clearance. The four-update native run is a separate diagnostic, not the starting
checkpoint of that eventual cohort.
