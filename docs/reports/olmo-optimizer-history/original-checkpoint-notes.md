# Original OLMo-1B checkpoint and public loss references

Checked 2026-09-29 with public HTTP metadata/small source and W&B GraphQL requests,
without credentials, checkpoint downloads, tensor loading, or GPU execution.
These are original **OLMo-1B** references, not OLMo-1B-0724 or OLMo2.

## Optimizer state: historically documented, not currently verified accessible

The [official historical CSV](https://github.com/allenai/OLMo/blob/ae84d479fa5775b1935b50b2120e0b514313ce18/checkpoints/official/OLMo-1B.csv)
maps step 60,000 to the `olmo-small/sw58clgr/step60000-unsharded/` directory.
The [same revision's README](https://github.com/allenai/OLMo/blob/ae84d479fa5775b1935b50b2120e0b514313ce18/README.md#checkpoints)
describes raw `model.pt`, `optim.pt`, `train.pt`, and `config.yaml` files.

Exact historical object URLs:

- [Optimizer](https://olmo-checkpoints.org/ai2-llm/olmo-small/sw58clgr/step60000-unsharded/optim.pt)
- [Model](https://olmo-checkpoints.org/ai2-llm/olmo-small/sw58clgr/step60000-unsharded/model.pt)
- [Trainer state](https://olmo-checkpoints.org/ai2-llm/olmo-small/sw58clgr/step60000-unsharded/train.pt)
- [Configuration](https://olmo-checkpoints.org/ai2-llm/olmo-small/sw58clgr/step60000-unsharded/config.yaml)

Default Python requests received Cloudflare 403 (`error code: 1010`). A bounded
browser-User-Agent HEAD check of `config.yaml` and `optim.pt` returned 404. The
possible `https://storage.googleapis.com/ai2-llm/olmo-small/sw58clgr/step60000-unsharded/`
counterpart returned anonymous `AccessDenied`, which expressly does not establish
whether the object exists. No working official mirror was found in this bounded
check; this is not proof that no retrievable copy exists anywhere.

The original CSV was [removed on 2024-11-26](https://github.com/allenai/OLMo/commit/889aaaa523872d6a2d1e4c0b017bf2fe2654c80c)
with commit message “Checkpoints aren't ready anyways.” Thus today's generic
checkpoint documentation must not be treated as verification of this particular
historical optimizer's availability.

The [selected HF revision's file inventory](https://huggingface.co/api/models/allenai/OLMo-1B/tree/81b71efbce6f4dada57c94860301af4298bcd351)
contains weights/tokenizer/config files and no optimizer or trainer state. Our
new Adam history therefore cannot be described as original pretraining history.
Before any future import, verify complete object bytes, model correspondence,
parameter mapping, per-parameter steps/moments, optimizer groups and precision.

## Public original loss near step 60,000 is accessible

The model card and official README link the [OLMo-1B report](https://wandb.ai/ai2-llm/OLMo-1B/reports/OLMo-1B--Vmlldzo2NzY1Njk1).
Its public project is `ai2-llm/OLMo-1B`; the old `ai2-llm/olmo-small` project was
not exposed by the anonymous GraphQL query. The published segment covering this
checkpoint is [OLMo-1B-run-003, ID `jis94ivf`](https://wandb.ai/ai2-llm/OLMo-1B/runs/jis94ivf).
It ends at step 60,435. Browser HTML is only an application shell, but anonymous
POST requests to `https://api.wandb.ai/graphql` returned its config and history.

| Published `_step` | `train/CrossEntropyLoss` | `train/Perplexity` |
| ---: | ---: | ---: |
| 59,999 | 2.600386142730713 | 13.468937963464805 |
| 60,000 | 2.598163604736328 | 13.43903597855462 |
| 60,001 | 2.577411651611328 | 13.163023537677185 |

Reproducible small query: request `project(name:"OLMo-1B",entityName:"ai2-llm")`
→ `run(name:"jis94ivf")` → `sampledHistory(specs:$specs)` with the JSON-string
specification below. The response returned all 22 steps in this interval;
`minStep`/`maxStep` are required—`start`/`end` did not constrain the sampled range.

```json
{"keys":["_step","train/CrossEntropyLoss","train/Perplexity"],"xAxis":"_step","minStep":59990,"maxStep":60011,"samples":100}
```

Published config: width 2048, 16 layers/heads, tied embeddings, RoPE, no Q/K norm,
T2048, global batch 2048, `amp_bf16`, mixed FSDP, no Flash attention. AdamW has
betas 0.9/0.95, nominal LR 0.0004, weight decay 0.1 and cosine warmup 2000 steps.
This nominal LR is not the instantaneous LR at step 60,000. The copied run config
also records `eos_token_id: 0`, whereas the later official config/current selected
native tokenizer contract use 50279. Do not change our tokenizer from that stale
metadata: matching the actual prepared token stream is the relevant authority.

## What a valid comparison would require

1. Use the ordinary original checkpoint without RT, FBT, or NextLat. Match the
   tokenizer, complete output vocabulary, actual Dolma-v1.5 mixture and chunk
   construction. Our small readiness corpus/first books prefix is not that mix.
2. Match T2048 and ordinary continuous causal packed attention. Internal EOS and
   the following document's first token remain ordinary CE targets; the last
   position does not predict into the next chunk. T1024/short isolated fixtures
   answer a different conditional-prediction question.
3. Match the denominator. The [pinned historical trainer](https://github.com/allenai/OLMo/blob/ae84d479fa5775b1935b50b2120e0b514313ce18/olmo/train.py#L608-L672)
   shifts labels within the chunk, sums its T−1 targets, then divides training
   CE by B×T **input tokens**. Our reports divide by valid CE targets. For full
   T2048 rows only, multiply that historical logged loss by 2048/2047 to express
   it per target: step 60,000 becomes approximately **2.599432859**. This source
   documents the retained historical implementation; exact original logger
   revision/aggregation would still need checking for a strict reproduction.
4. Distinguish pre-update training loss from saved post-update weights. The
   historical trainer computes the loss, applies Adam, and logs that pre-update
   loss at the incremented step. A checkpoint saved afterward is not expected
   to reproduce the same number exactly even on that same batch. Matching the
   data-order file and relevant batch/weight boundary would be necessary.
5. Treat the public curve as a rough sanity reference unless all of these match.
   For a clean precision diagnostic, evaluate the same frozen native weights on
   the same held-out tokens in FP32 and BF16 with the same loss implementation.
   Compare ordinary fidelity first; do not attribute a mismatch on different
   data/context to precision, RT, or optimizer history.

No new model evaluation or training was performed for these notes. The separate
optimizer-history diagnostic uses our saved continuation moments and explicitly
fresh moments at the same model state; it does not recover original OLMo Adam.
