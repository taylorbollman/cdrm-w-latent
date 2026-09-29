# Ordered pilot execution progress

2026-09-29: in progress on `feat/olmo-pilot-execution`, starting at `3d5c949`.
The user authorized ordered-runner integration, two-GPU restart/evaluation
acceptance, then bounded native T1024 capacity measurements. No quality
campaign or precision sweep is planned. Both H10080GB devices were verified
inside the required project container and idle at entry. SSD has ~1.4TiB free;
boot has ~91GiB free. All new large checkpoints belong on SSD and must be
retained to GCS; evidence stays under `.runtime/olmo-pilot-execution/`.

Root owns `olmo_pilot_execute.py`, runtime launch, native declarations, tracking,
retention and closeout. Agent pilot_contract owns new declaration/resolution
module/tests; pilot_eval owns named dev evaluation module/tests; pilot_acceptance
owns tiny fixture and independent audit modules/tests. No pre-existing runtime,
model, vendor or test file may change. Shared engine is PR48's unchanged SSD
engine. Native construction reuses its accepted model constructor. Source pins
must be frozen before GPU acceptance and kept stable through resume.

New protocol: [protocol.md](protocol.md). Tiny schema is
`olmo-pilot-execution-tiny-acceptance-v1`; native schema is separately versioned.
Tiny T16 / B2 per rank / NFR / three updates of80inputs includes true document
boundaries and uneven rank slots. Compare reference with named FP32 dev panels,
then stop2/cloud restore/resume3 and terminal evaluation-only acceptance.

Evidence namespace for this milestone:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T190649Z`.
Checkpoint namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T190649Z/pilot-execution`.
No GPU run started yet. Runtime/CPU code and test development are ongoing.


19:24 UTC: runtime is frozen at `be74dde`, 192 source pins with canonical digest
`4d6f9fd0d3e61e45e13236d2c2804913b9a4790275b3e71c37657df2da83a7cf`.
PR50 is open in draft. New source/test suite passed149distincttests in58.75s.
CPU evidence retained in `pilot-execution-cpu`; synthetic fixture corpus retained
separately with `olmo_document_retain.py`. Native B32/NFR12/B64/NFR8 CPU resolution
completed in native-declarations-01; these are not launched yet.

Tiny reference and evaluated runs completed. Insertion audit passed2063checks,
all3updates rawgradients/completeboundaries exact on bothranks. Evaluated dev-main
and books separately at update2 with independent B1/rank FP32 evaluation. Tiny
stop2 completed/retained; `tiny-restore2-01` is restoring its exact cloudgenerations
for resume3 and terminalstop2 checks. AllGPUlaunches are rootowned and sequential;
inspect launch-result JSON and report before starting another. Commands and logs
are in .runtime/olmo-pilot-execution; launch.py enforces external timeout.

Small additional adapters were needed for the newidentity: checkpoint storage
subclass overrides only two metadata validation methods; restore uses explicit
newidentity validators with unchanged streaming and bytepublication helpers.
Old outer cursor/checkpoint schemas remain shared, ordered inneridentity is new.
No preexisting source/test changes. Follow sourcefreeze through all resumes.
