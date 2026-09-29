# Packed campaign readiness progress

2026-09-29: started `feat/olmo-packed-campaign` from `7b9c614`. Two idle H10080GB
GPUs verified inside the required container; retained Dolma token manifest is
present on SSD. No retokenization or quality run. Frozen policy/protocol in
`protocol.md`; runtime/model policy, disk-backed data and bounded component
precision probe are being implemented independently. Root owns GPU launches
and actual-data training/restart integration. Preserve progress every 20–30 min.

2026-09-29 resume: no old GPU processes were left running. Explicit stream
policy/model/loss changes and bounded per-component precision probe are committed
in `b9985bc`. Disk-backed index and real-data restart runner are implemented.
Independent model, data and runner reviews found no blocking issue; a narrow
index-file verification race is being hardened before runtime freeze. Initial
scoped CPU passes: model 378 plus one focused boundary case, data 18, precision
13, runner/probe integration 33 (overlapping scopes). Combined regression is
running; first launch only failed collection due a nonexistent test glob and
ran zero tests. No new GPU acceptance claim yet.

Evidence root: `.runtime/olmo-packed-campaign/`. New cloud namespace:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T023500Z/`.
Upcoming order: tiny packed eager/graph, pretrained packed prepared reference,
bounded isolated CE/latent/KL precision localization, actual packed T1024 write,
cloud retain/restore, then fresh-process resident-Adam cold capture/resume.
The real update has 524,288 valid inputs, 512 chunks; B12/rank on two GPUs needs
22 accumulation slots/rank with padded final slots. All qualification failures
remain separate from operational readiness. No quality training queued.
