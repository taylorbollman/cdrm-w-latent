# Context choice and future RT screening intent

2026-09-28. User places FA4 aside and is considering context 1024 if it recovers
enough of the combined model's T512 throughput while providing more context.
They are most skeptical about RT's marginal value and envisage an early screen,
possibly approximately 500 million continuation tokens plus SFT. That budget,
dataset, evaluation and training schedule are provisional. This milestone only
benchmarks execution; it does not launch the screen.

A useful primary quality comparison is matched **FBT + NextLat with and without
RT**. Preserve the starting pretrained checkpoint, continuation data/order,
context and loss masks, training/evaluation recipe and token budget. This isolates
the contribution the user most wants to assess. An ordinary OLMo continuation
can anchor the absolute value of the overall recipe; a full eight-combination
campaign need not precede a directional RT decision.

Track both tokens seen and actual compute/wall time. Equal-token results measure
learning differences, while the extra RT cost matters for choosing what to keep.
The no-RT control's throughput is not established by ordinary-only throughput:
FBT and NextLat still perform additional work.

Define the held-out tasks and practical continuation criteria before inspecting
the quality result. Include tests whose dependencies fit the selected context.
A 1024-token screen would not settle whether longer-context FBT or RT is useful.
An early negative result can justify deprioritizing RT for this project under
the tested recipe; it would not demonstrate that RT has no value generally.
Check adaptation is functioning before interpreting a failure as lack of useful
capacity. One budget/seed is a screening decision, not a definitive benchmark.

The compute-only 500M-token estimate in results.md uses the current synthetic
benchmark's supervision and steady complete-update rate. In particular KL uses
response-half positions, while CE and latent supervision cover all valid pairs.
Full-sequence KL would add work. Startup, data loading, evaluation, checkpoint
I/O, packing behavior and SFT are excluded. Validate the final data/objective
configuration briefly before committing to a long-run time estimate.

Existing BF16 qualifications remain recorded. Passing own eager/graph checks
does not resolve independent precision comparisons or establish optimization
equivalence. Avoid reopening broad numerics solely because a quality screen is
being planned; address concrete new failures if they occur.
