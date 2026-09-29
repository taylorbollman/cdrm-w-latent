# Ordered execution acceptance and capacity operations

The new executor preserves the shared SSD update engine. Its report and
execution identity have new ordered-stream schemas; the checkpoint container,
SSD configuration and rank cursor wrapper retain their accepted schemas. The
inner cursor binds the ordered panel manifest. An old packed checkpoint is not
an ordered-stream restart authority.

Use fresh output and SSD segment directories for every attempt. Keep the
checkpoint manifest and producer receipt byte pins. No GPU command runs in the
host shell. Before each GPU queue, enter the project container, assert
`/.dockerenv`, check `/workspace/cdrm-w-latent`, and run `nvidia-smi` there.
All startup/source authorities must be frozen before capture. A source change
invalidates the same-lineage reference/resume pair.

## Tiny fixture

`python scripts/olmo_pilot_execution_fixture.py --output-dir NEW` is CPU-only.
It uses the actual pinned native OLMo tokenizer and creates synthetic short
text, true internal EOS, document shards and all 21 real ordered indexes. The
raw source URLs/cloud generations are explicitly simulated test metadata, not
Dolma acquisition evidence. `report.json` contains the complete `data_spec`.
Three T16 logical updates consume 80 valid inputs each, five rows distributed
across two ranks with physical B2. Each rank makes two calls; three physical
rows per global update are dummy padding. Cross-document CE is present while
NextLat pair/triple masks respect true documents. The 27-row train panel leaves
headroom beyond the 15-row acceptance schedule.

The two declared diagnostic dev prefixes are `dev-main` (80 valid inputs) and
`dev-source/books` (32 valid inputs). Routine confirmation evaluation remains
prohibited. Main/source overlap is reported separately. These synthetic tests
exercise allocation and restoration; their losses carry no quality meaning.

Run a no-evaluation reference, a matching evaluation-inserted reference, a stop
at update 2, and a fresh-process cloud-restored continuation to update 3. The
independent JSON audit compares exact update input/noise, raw gradients, loss
metrics, model/Adam/schedule/RNG/cursor state and named evaluation results. It
also verifies SSD publication order, keep-two pruning and producer receipts.
Cloud download byte checks and actual checkpoint loading are separate evidence;
a JSON audit does not replace either.

`olmo_pilot_execution_restore.py` is required for the new identity. Historical
restore validators intentionally reject it. Never patch a historical schema or
silently cast a new checkpoint into an old execution identity.

## Bounded native resource check

Only after tiny acceptance, measure the ordered reader and unchanged model
with actual T1024/BF16 graphs, activation checkpointing and resident Adam.
NFR means K4 FBT, native RT at layers 0 and 15 on all passes, and both NextLat
losses. Keep original startup versus fusion-only adapted startup explicit;
these disposable capacity runs are not a matched quality experiment.

The relevant prior authority is
[the campaign two-GPU report](../olmo-campaign-two-gpu/results.md), not the older
K2 fixed-layout benchmark. Its isolated full-valid K4/T1024 probes measured:

| Physical batch per GPU | Valid input tokens/s | Peak reserved per GPU | Sampled free per GPU |
| ---: | ---: | ---: | ---: |
| 8 | 3,428 | 50.35 GiB | 22.97 GiB |
| 12 | 4,311 | 59.06 GiB | 14.23 GiB |
| 16 | 4,970 | 69.97 GiB | 3.30 GiB |

The frozen protocol starts NFR at B12/rank, with B8/rank as the fallback if
setup or evaluation lacks comfortable headroom. Do not pursue B16 to consume
the final few GiB. Ordinary B starts at B32/rank; consider B64/rank only after
the measured margin supports it. Old T512 ordinary rates are not a T1024
campaign capacity acceptance. These are bounded candidates, not promises of
fitting the actual ordered execution path.

Use a fresh process per physical batch. Each capacity declaration has eight
updates: exclude updates 1–3 from timing and report updates 4–8. Keep capture
warmup, checkpoint write/cloud transfer, held-out evaluation and source audits
outside the compute-region update interval. Report global valid input tokens once, actual CE/
latent/KL counts, physical rows/rank, accumulation, setup peak allocated/reserved,
steady reserved and sampled free memory. A large logical batch obtained through
accumulation is not a large RT kernel batch.

The capacity declarations use a 600-second checkpoint policy plus checkpoints
every four completed updates, including retained origin and final boundaries.
This is a bounded test setting, not the selected cadence for a future learning
run. Choose that cadence from measured write/transfer costs while retaining
progress within the intended approximately 20-minute interruption window. Time
policies are checked at completed update boundaries; an indivisible longer
operation must be reported separately. The present declaration validator caps
the configured time interval at 600 seconds, so a larger interval would require
an explicit policy change before launch.

Initially allocate common-FP32 dev evaluation at B1/rank independently of the
training batch. Measure its cost and memory before choosing routine monitoring
membership and cadence. A two-H100 resource result is not H200 capacity or
cross-topology restart acceptance. No quality token budget follows from these
measurements automatically.
