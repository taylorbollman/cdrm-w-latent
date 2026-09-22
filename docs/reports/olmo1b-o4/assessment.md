# O4 assessment and next action

Reviewed 2026-09-22. All four arms completed 2,634 updates / 20,855,799
valid input tokens without a restart or health-gate stop. Their final full
optimizer checkpoints and the source/data/result evidence have verified GCS
receipts. See [results](results.md) for endpoint metrics, paired document
intervals, plots, W&B links and checkpoint hashes.

Ordinary continuation is the best endpoint on both measures. Code NLL falls
from 1.78790 to 1.69936. RT reaches 1.70635: the gradual layer-0 conversion
recovers most of the ordinary control's code quality, despite the unadapted
alpha-one preflight NLL of 3.54586. Its remaining +0.00699 code-NLL difference
is small but consistent across sampled evaluation documents. Retention costs
are larger: NLL 3.24892 for RT versus 3.17382 for ordinary and 3.04099 before
continuation. This is a conversion/recovery result, not an RT advantage.

NextLat costs approximately 0.093 code NLL in either architecture. Its code
interaction with RT is -0.000697 nats, with a document-bootstrap interval
crossing zero. There is no evidence here of the useful RT/NextLat interaction
seen in some earlier synthetic settings. These intervals describe evaluation
documents, not variability across training seeds.

The curves help localize the problem. At update 50, **before RT activation**,
both NextLat arms have code NLL about 2.176 on the 128-window curve subset,
versus about 1.645 without NextLat. The first training batch has CE 1.738,
latent loss 0.996 and KL 11.49 at unit auxiliary coefficients. Thus a randomly
initialized predictor imposes a large new objective on a pretrained backbone.
By the final 100 updates, mean KL is about 0.543 and latent loss about 0.116,
yet the code deficit remains and retention continues worsening. This suggests
an auxiliary-objective adaptation problem; it does not prove its full cause
or show that longer training could never help.

Numerical health is distinct from quality. Every recorded loss and gradient
norm is finite. **All 2,634 updates in every arm are clipped at norm 1.**
Median pre-clip norms are 2.435 / 3.291 / 2.553 / 3.433 for ordinary,
ordinary+NextLat, RT and RT+NextLat; maxima are 5.102 / 47.905 / 7.972 /
47.883. Describe this as controlled under clipping, not as gentle unclipped
optimization. No new precision defect is indicated by these records.

Continue with a bounded **O5a FBT implementation/correctness milestone**,
starting from the same original checkpoint. Preserve ordinary pass zero,
shifted previous-pass feedback, independent RT/FBT/NextLat controls, attached
cross-pass gradients and the planned pass-loss reduction. Check finite Jacobi
passes against an independently constructed reference and exact online
execution on short sequences. Keep the initial FBT learning control free of
NextLat when that learning protocol is subsequently selected.

Do not automatically extend the four O4 arms or queue another large learning
run. Before another pretrained NextLat learning comparison, a bounded
predictor-only warm start and/or independently ramped auxiliary coefficients
is a better diagnostic candidate than simply repeating the initial shock.
These remain proposed experiments, not changes to the completed O4 recipe.

Limits: one seed, bottom RT layer only, short code continuation, development
subsets, no programming execution score, unknown pretraining overlap, and
general-language retention on independent windows. Reserved tests remain
untouched. FBT is a separate hypothesis; this pilot does not veto it.
