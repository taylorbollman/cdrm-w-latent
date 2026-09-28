# Combined T1024 evidence retention

2026-09-28. All three new stages and the separate closeout bundle are verified
in GCS. No failed/incomplete run is presented as passed.

Prefix: `gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T181800Z/`.
The retention helper's legacy prefix does not mean two GPUs were used.

Each stage contains evidence.tar.gz, retention-manifest.json and a cloud
storage receipt. Archive/manifest objects were checked for server size, MD5,
SHA256 metadata and downloaded SHA256. Local receipts are under
`.runtime/olmo-combined-t1024/retention/`.

| Stage | Members | Archive bytes | Archive SHA256 |
| --- | ---: | ---: | --- |
| `sdpa-b32-01` | 165 | 670,796 | `1e729a0d2ec5117f02c8945132480598b25eaa5060a40ce8e280e7862f771c01` |
| `sdpa-b64-01` | 165 | 670,755 | `d42935bda2c5b3b9092c8eba475d8c1c854070ad5fe9b24311b637b01fab13eb` |
| `sdpa-b64-02` | 165 | 670,744 | `951a39d624afef808b28436562d2d827df2adb5eea18a01a84291e74bd0f83fe` |

The final audit verifies 489 new source snapshot pairs, 15 operational gates,
24 physical updates and three retained stages. All 37 pretrained source files
and recorded dependencies match the retained references. No FA4/Dao external
source snapshots are required by this SDPA/native-RoPE scope; package versions
remain in every report.

Historical T512 reference:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260925T172000Z/single-combined-b128-01/`.
Historical T2048 reference pair:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260928T165900Z/sdpa-b32-01/`
and `sdpa-b32-02/` under that same timestamp. Their reports, sources and saved
retention receipts were re-audited without new GPU execution.

Closeout includes derived JSON/CSV/plots/helper, CPU/process/final-GPU-state logs,
three stage receipts, and copies of the three historical reports with frozen
sources and receipts. It is not another execution stage. No credentials, W&B
cache or model weights enter this bundle. Original O1 weights remain retained;
no new full checkpoint is needed for disposable timing updates.

Closeout verified: 517 members.

| Object | Bytes | SHA256 |
| --- | ---: | --- |
| `evidence.tar.gz` | 2,199,202 | `1ce1b5c9f0dd412c3ebb9ded291e651cf875d2528ac210d534fe3762b08ac637` |
| `retention-manifest.json` | 137,316 | `bc1681c4ea36669553992479660382870032c1211b0a2441f6eb0d779ca7ec4a` |
