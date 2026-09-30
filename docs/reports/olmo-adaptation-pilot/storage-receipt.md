# Retained evidence and recovery states

All three update32 endpoints and every submitted cadence checkpoint are
cloud-verified. States remain under the immutable declaration cloud roots,
with unique segment suffixes; no prior run was overwritten. Local SSD keeps
the newest two verified states per owned segment and may disappear with the VM.
Use verified cloud authorities for recovery.

Checkpoint root:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T204500Z/pilot-async/`

| Arm | Suffix below root | Verified updates | Final state SHA256 |
| --- | --- | --- | --- |
| B | `b/B/native-b32-first32-01/` | 0,32 | `8b72a2d7f505a7078b3d2a7a4ebf1f0f8ddb6c02e0b8d9bc635c3e2198100847` |
| NF | `nf-nfr/NF/native-nf12-first32-01/` | 0,8,17,26,32 | `890c235221252e8fe2144817796fe77d09b9099c276f4a0749d8d9880c8d974c` |
| NFR | `nf-nfr/NFR/native-nfr12-first32-01/` | 0,2,7,12,16,21,26,31,32 | `950ff9396a87ca63a92148b810386e8b748ab8cc4f765b851ebc4d79e82c5369` |

Each boundary directory is `update-NNNNNN/` with `state.pt` and `manifest.json`.
Final B state is14,155,077,221bytes; NF/NFR each15,214,966,145bytes. Manifests,
server generations/MD5, size and downloaded SHA256 verification are recorded
in the native reports and inventory; these hashes identify the saved objects.

Small evidence prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260929T231346Z/adaptation-pilot-`

Verified stage suffixes include each native stage, `summary-cohort-01`,
`tracking-cohort-01`, `queue-02`, declarations, launch/recovery snapshots,
analysis helpers, paired origins and progress snapshots. Persistent receipts
are `.runtime/olmo-adaptation-pilot/retention/<stage>.json`. The preserved
queue01 evidence is included in recovery input snapshots; queue02 is the
completed scheduling authority. W&B charts supplement these artifacts.

Inventory `.runtime/olmo-adaptation-pilot/inventory-01/report.json` has SHA256
`43cad4ec033ac479ae1bf7e74d64e6b2b85df10330c8659a5af62bb88cf16bd5`.
Its immutable receipt snapshot accounts for:

- 16 unique checkpoint states:211,619,293,462bytes.
- 16 checkpoint manifests:9,520,366bytes.
- 21 small evidence archives/manifests:53,588,454bytes combined.
- 74 distinct objects totaling211,682,402,282bytes.

The inventory deduplicates repeated object references, validates publication/
journal/latest authorities, and copies receipt bytes under neutral names so
archive exclusions for `checkpoint-` paths cannot silently omit them. It reads
local receipt metadata, not state payloads, and performs no new cloud readback.
The full upload/readback verification already occurred in the runtime/retainers.
Its own later upload and later closeout/admin receipts are outside this snapshot
by construction; see [progress](progress.md) for those receipts and merge state.
No new corpus or model-weight download was needed in this milestone.

The600-second checkpoint trigger is not a guarantee of at most ten minutes of
lost work. Save-boundary delay, synchronous local save and background verification
extend that interval. Until a new publication is verified, the preceding cloud
checkpoint remains authoritative. Final retention drains before closure.
