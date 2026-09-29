# Pilot progress

2026-09-29: user authorized the proposed first32 comparison. Branch
`exp/olmo-adaptation-pilot32`, starting main `358ab65`. Container preflight
confirmed two idle H10080GB GPUs, about90GiB boot free and1.3TiB SSD free.
The prepared declarations were copied byte-for-byte into
`.runtime/olmo-adaptation-pilot/declarations-01`. CPU-only async preflight
passed for B/NF/NFR; all200 runtime pins unchanged. Adoption report SHA256:
`6e9fc86876a582ebefa2cd1ed62ce36408ce5ea73a6c86dbb567c3a3d7d2ce1a`.
All128 ordered memberships/counts remain equal. First32 counts per arm are
16,777,216 inputs and16,760,832 CE targets; enabled auxiliary counts for NF/NFR
are16,730,817 latent pairs and16,684,487 KL triples. Disabled B losses count0.

Queue helper `.runtime/olmo-adaptation-pilot/run_queue.py` runs B, NF, NFR
sequentially, with explicit lean/async/stop32 and14400s external bound per arm.
It stops scheduling after a failed/incomplete stage. Stage names:
`native-b32-first32-01`, `native-nf12-first32-01`, `native-nfr12-first32-01`.
Expected report status at32 is `stopped_at_boundary`, segment_completed=true,
plan_completed=false. Keep the128-update finite plan unchanged. New SSD
namespace is `adaptation-pilot`; old cloud declaration roots gain these new
unique segment suffixes. All GPU work remains inside the required container.

Root owns launch/monitoring. The `native_async_declaration` agent independently
audited launch and is preparing a JSON-only result summarizer; no other GPU
jobs are authorized. Inspect `queue-01/report.json`, individual reports and
process state before recovery. Do not repeat completed work after interruption.
