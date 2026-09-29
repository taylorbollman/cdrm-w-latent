# Fixed-generation checkpoint readback results

The CPU readback comparison passed: both methods verified the same complete **14,154,933,413-byte** checkpoint against its pinned SHA256 and MD5. Streaming avoided the whole-object host-memory allocation. This one ordered pair shows essentially unchanged elapsed time; it does not establish a speed improvement or change the current retainer.

| Measurement | `download_as_bytes` | Stream to hash/count sink |
|---|---:|---:|
| Complete download and Python SHA256/MD5 verification | 92.308 s | 91.604 s |
| SDK download call | 61.201 s | 91.604 s, including sink hashing |
| Python verification after download returns | 31.108 s | 0.000018 s |
| Separate metadata/client lookup | 0.658 s | 0.679 s |
| Recorded peak RSS | 13,891,440 KiB (**13.248 GiB**) | 113,068 KiB (**110.418 MiB**) |
| Peak RSS before download | 113,068 KiB | 113,068 KiB |
| Increase in recorded high-water RSS | 13,778,372 KiB | 0 KiB |
| Current RSS before download | 68,608 KiB | 68,128 KiB |
| Current RSS after verification | 13,893,268 KiB, payload still live | 68,184 KiB |

Fresh Python child processes isolate the two payload runs, but their high-water RSS was already ~110 MiB before downloading. It is not a zeroed allocation meter: process-creation/exec history and startup/import allocations can contribute to the pre-existing high-water mark. The stream's zero increase means it stayed below that mark, **not** that it used zero memory. Its current RSS rose only 56 KiB between the two snapshots. The bytes arm's current RSS is ~1.8 MiB higher than its recorded peak; these are separate Linux `/proc` and `getrusage` accounting/sampling observations, not an exact instantaneous memory trace. The ~13 GiB whole-payload difference is clear despite that minor discrepancy.

The streaming SDK delivered **1,727,898 writes**, each at most **8,192 bytes**. Its SHA256/MD5 computation ran inside those writes; comparing its 91.604-second download call to the bytes arm's 61.201 seconds would omit the latter's additional 31.108 seconds of hashing. Both complete totals include Python MD5 as well as SHA256; the old retainer's post-download Python pass computes SHA256 only, so these totals are not its exact existing readback cost.

## Scope and controls

- Root ran one pair in fixed order, bytes then stream, after the earlier checkpoint transfers had completed. Both children exited 0; the parent completed successfully and [W&B synchronized](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/4zk92f7f).
- No GPU devices were exposed to the container, neither child imported PyTorch, and no tensors were loaded. There were two full readbacks and no uploads or local checkpoint copies.
- Parent PID 96 and child PIDs 176/190 were distinct. Python, platform, `google-cloud-storage` **3.12.0**, and installed SDK source hashes matched. Both children authenticated the completed reference report and its local/published checkpoint-1 receipt chain.
- Remote metadata had no content encoding, default `chunk_size=None`, and the exact generation, size, MD5 and SHA256 metadata. Full final byte count, SHA256 and MD5 were independently checked in both arms. The sink rejects overflow or seeks rather than attempting a rewind of its hash state.
- Default SDK range-retry behavior remained enabled. The focused CPU suite passed **29 tests**, including actual SDK `Download.consume` logic with a synthetic interrupted response and append-only range retry, plus fail-closed transcoding rewind. This live pair did not deliberately inject a network fault; it does not independently establish that a real retry occurred.

One ordered pair cannot distinguish cache, network and scheduling effects. SDK/client/authentication setup was outside the transfer timer. The observed difference in complete times is only 0.704 s (~0.76%); **no general speed claim is made**. The result establishes a useful memory property for this pinned object and environment, not upload performance, asynchronous checkpoint safety, or end-to-end checkpoint pause time.

## Authority and independent audit

The retained object is:

`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/base-loop-base-reference-01/update-000001/state.pt`

| Authority | Value |
|---|---|
| Generation | `1790680957093316` |
| State SHA256 | `5ed5a23f7d9f2753e76bbf95ed45797f8ddafb30f82ca7b276dc8e98ce2203e5` |
| State MD5, base64 | `oNJ04OABVkvef0iAR49pVA==` |
| Checkpoint-manifest SHA256 | `748bb7a190c2d136119d4c6924b9ab6324b1bf3041c2903bae108ea55a5f7718` |
| Original reference-report SHA256 | `6e945cf271c1791a3dcb4e71f2f481e8389bb59477dad1f78097db2a23563d25` |
| Parent readback-report SHA256 | `0401d4b7c89d992109d5362c8ecb8c793f7aee92d8687bc8ad2b68e55d8f4f44` |
| Bytes child-report SHA256 | `21fb0592cd8744aa734c371d14d7db52cf86f03ac1d03b6a56fb41433ad5a189` |
| Stream child-report SHA256 | `ced24122cd016a3b535c556641051de773d072a9f555b93141e7f17b73132fcd` |

Reports and source snapshots are under `.runtime/olmo-campaign-lifecycle/readback-01/`. Frozen implementation commit: `dfacdb0`; [protocol](readback-protocol.md). A separate stdlib audit passed **47 checks**, including **18 live-source/report/snapshot matches**, exact embedded child reports, shared authority, environment and digest identity, timing arithmetic, and bounded stream chunk counts. It performed no additional transfers. Its script/report are in `.runtime/olmo-campaign-lifecycle/readback-audit-01/`; audit-report SHA256 `809fc6aae99337d3eae87ba95f8649375cc72a48227bd269de01e3888387a218`.

## Practical consequence

This supports using a separately reviewed streaming verification path in a future operational retainer to remove roughly one checkpoint-sized host allocation while keeping full readback verification. The immutable state-before-manifest publication rule, generation checks, final full digests and RNG isolation must remain. **No existing helper, default or retention policy changed in this experiment.**

Another readback pair is not necessary to establish this directional memory result. The more useful next efficiency measurement is direct timing of local serialization, upload, readback and checkpoint publication in an operational run, as proposed in the [checkpoint-cost assessment](checkpoint-cost-assessment.md). Those components remain insufficiently separated in the earlier acceptance results.
