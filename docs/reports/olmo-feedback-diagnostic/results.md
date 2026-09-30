# Saved-checkpoint feedback diagnosis

[W&B charts and tables](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/p5xs1bod)
and [standalone figure](feedback-diagnostic.pdf).

The bounded checks found a substantial feedback/adaptation problem, without
finding an execution-control or gradient-decomposition defect. The later-pass
deficit already exists at the saved starting checkpoint. Reducing feedback to
half strength does not repair it. Auxiliary agreement improves while language
prediction remains poor, so falling latent/KL losses should not be interpreted
as successful refinement.

NF means four-pass FBT plus NextLat, without RT. NFR adds native RT at layers
0 and 15. Beta controls how the fusion output replaces the original embedding;
beta 1 is full replacement, beta 0 bypasses fusion, and beta 0.5 interpolates.
All observations below use fixed saved weights, FP32, no jitter, and T1024.
There are no optimizer updates or changes to the trained checkpoint bytes.

| Same eight dev sequences | Pass 1 CE | Pass 2 | Pass 3 | Pass 4 |
| --- | ---: | ---: | ---: | ---: |
| NF origin, full feedback | 2.674 | 7.364 | 7.158 | 7.188 |
| NF update 32, full feedback | 2.947 | 7.348 | 7.390 | 7.401 |
| NF update 32, half feedback | 2.947 | 7.433 | 7.437 | 7.439 |
| NF update 32, feedback bypassed | 2.947 | 2.947 | 2.947 | 2.947 |
| NFR update 32, full feedback | 3.032 | 6.974 | 7.010 | 7.017 |
| NFR update 32, feedback bypassed | 3.032 | 3.032 | 3.032 | 3.032 |

Bypass equality is exact in hidden-state bytes, executed inputs and losses for
every row; first-pass results also remain exact across feedback strengths.
NFR reproduces the earlier mixed signal: somewhat better later-pass CE than NF,
but worse first-pass CE, and still no useful refinement over its own first pass.
The eight-row subset differs from the cohort's larger panel. Improvement from
updates 16 to 32 does not imply improvement from the actual origin.

At NF32, later-pass teacher entropy is about 7.66–7.70 nats and predictor entropy
8.01–8.03, while KL has fallen to about 0.32. Their readout distributions have
become broad and closer to each other. Within-row positional variation also
falls. These observations are consistent with an easier auxiliary prediction
problem; they do not prove collapse or establish which loss caused the deficit.
See [forward analysis](forward-notes.md) for masks, aggregation and geometry.

The gradient analysis separates first-pass CE, later-pass CE, latent regression
and KL using the existing objective and detach semantics. On the two NF batches,
auxiliary gradients oppose first-pass CE on the backbone but align strongly
with CE on fusion. Combined CE/auxiliary backbone alignment changes sign across
batches (-0.058 and +0.231). The total raw gradient has positive dot products
with both CE contributions on both batches: this is selective competition,
not a demonstrated reversal of CE learning. NFR shows stronger opposition:
combined CE/auxiliary cosine -0.536 on the backbone and -0.646 on fusion.
Its joint-gradient dot products with both CE contributions remain positive.
The endpoints have different trained weights, so this comparison does not isolate
RT as the cause. See [gradient analysis](gradient-notes.md) for exact norms.

All three gradient decompositions pass, with global reconstruction relative L2
at most 2.47e-6 (0.000247%). Weights, buffers, RNG and gradient ownership remain
unchanged across all six saved-state probes. These checks validate the bounded
decomposition and controls, not every possible numerical execution path.

These local gradients are not the large-batch BF16 Adam update. The cohort still
has fresh Adam, a newly initialized predictor, only 32 of 100 warmup updates,
one seed, and limited dev coverage. Existing BF16 qualifications remain intact.
The next discussion is a small training comparison, not another precision grid;
see [next steps](next-steps.md). No continuation has been launched.
