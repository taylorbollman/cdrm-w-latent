# Campaign two-H100 progress

2026-09-28: recovered current handoff and verified two idle H10080GB GPUs in
the required project container, NV18 connectivity, NCCL2.30.5, enlarged boot
disk and empty local SSD. Branch `feat/olmo-campaign-two-gpu` starts at `dde3240`.

Runner, distributed comparison probe and fresh-process restart harness are
being implemented independently. Root owns GPU launches. Initial NCCL sanity
stage is `.runtime/olmo-campaign-two-gpu/nccl-01`; see its launcher log/report.
No quality-training run has been launched. Planned protocol is recorded before
campaign GPU acceptance. Update this file and the results after each gate.

Initial NCCL sanity passed all5message sizes (4bytes through256MiB), exact sums;
W&Bci14andb. At100MiB median rank-max0.39675ms, at256MiB0.91291ms. These are
collective measurements, not model throughput.

Restored all86 finaltokenized objects fromGCS to the emptySSD, using recorded
generations and SHA256/size validation. Corpus verifier confirms28shards,
12,512unique documents and7,054,230tokens. Restore evidence is in
`.runtime/olmo-campaign-two-gpu/data-restore/`. No retokenization or packing.
