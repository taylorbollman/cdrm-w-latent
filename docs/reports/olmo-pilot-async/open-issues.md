# Open adaptation concern: clipping and later-pass CE

**Status: open, carried into the bounded adaptation pilot.** Execution and
checkpoint acceptance do not resolve this concern. Do not drop it from the
next milestone summary merely because all operations are finite.

PR50's combined K4 FBT + native RT + NextLat capacity fixture clipped every
update: raw gradient norm fell from 222.53 to 24.14 against a limit of 1.0.
Common-FP32 dev-main CE was about 2.993 on pass 1 versus7.654/7.652/7.631 on
passes 2/3/4. This was eight small-batch updates with adapted fusion and a fresh
optimizer; it was not a matched learning comparison.

PR51's four accumulated updates consumed 2,097,152 inputs. All were finite and
clipped: norms were 211.12, 76.50, 73.80 and 52.42. Aggregate training CE fell
from 7.740 to 5.666, but the larger fixed 65,536-input FP32 development prefix
returned pass CEs **3.18058, 7.87717, 7.71814, 7.74559**. Thus the concern
persists; successful execution and falling training CE do not establish useful
refinement. This development membership and exposure differ from PR50, so
their endpoint difference is not a learning trajectory.

The next evidence comes from normal pilot monitoring, not another general
BF16 investigation. Use the same fixed dev membership for NF/NFR, matched new
input exposure and identical imported fusion weights. Track every pass CE and
its difference from pass 1, the separate CE/latent/KL training terms, raw gradient
norm, clipping coefficient/frequency and the original-model continuation control.
The four-update accumulation diagnostic has a different finite plan and is not
a cohort endpoint or a measured pristine-start baseline.

At the first 32-update review, explicitly assess whether later-pass gaps close,
whether pass 1 degrades and whether gradient/auxiliary scales settle. Persistent
clipping by itself is not an automatic failure. If auxiliary terms improve while
CE worsens, or none of these indicators improves, pause extension and inspect
loss balance/fusion adaptation with a bounded saved-state per-loss gradient
probe. Keep initialization/exposure, data membership and precision fixed during
that diagnostic. Change a loss weight, LR or architecture only with a stated
reason and a separate labeled lineage.

The planned evaluations begin at update 16, not update zero. Thus first-pass
degradation means a subsequent trajectory or a difference against B at matched
new exposure; this pilot alone does not measure immediate pristine-start
retrofit damage. The four-update fixture's different development allocation
also prevents treating its CE difference from PR50 as a learning curve.

An actual nonfinite value, incoherent state/counter or failed integrity check
remains an immediate execution failure. Lack of useful refinement is a distinct
adaptation question; it neither proves an RT defect nor grants BF16 clearance.
