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
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-pilot-execution/20260929`.
No GPU run started yet. Runtime/CPU code and test development are ongoing.
