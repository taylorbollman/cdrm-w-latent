# Progress

2026-09-30 15:24 UTC: began the user-authorized endpoint probes and conditional
KL0.1 NFR64→128 milestone. Both H100s verified idle in the required project
container; no prior queue/training process remains. The completed NFR64 pair,
its populated checkpoints and cloud receipts are available. SSD has about
1 TiB free; persistent boot disk about 88 GiB. Large new states remain on SSD
with verified cloud retention.

Branch: `feat/olmo-nfr-stability-128`, from main `4b15854`. Implementation is
being prepared in separate, explicit diagnostic and continuation scopes.
See [protocol.md](protocol.md). GPU execution has not yet started.
