# Pretrained reference discrepancy: bounded localization

2026-09-29, after pretrained-eager-01. The frozen independent canonical gate
failed for NFR at the first distributed raw backward: relative gradient L2
0.0340224273. Both ranks agree on this value. CE sums are exact; latent sums
differ by1.335e-5, KL by8.583e-5, and objective by4.194e-6. All loss budgets pass.
Ordinary B passes all11 gates, including three Adam updates and replica parity.

This result alone does not distinguish DDP accumulation from BF16 changes
between sparse selected-position losses and the graph-compatible dense masked
loss layout. Tiny FP32 independent comparisons passed all8arms. Existing
one-GPU full-model evidence compared replay with the same prepared BF16 path;
that evidence does not resolve this new independent-reference miss.

Keep the failed attempt and its unchanged budgets. Add a separately identified
prepared local reference: world_size1, same physical rows/noise and global
denominators, no DDP, no graph. At identical initial weights, record canonical
versus prepared gradients/losses first as an explicit retained qualification.
Then compare real eager and captured DDP with the prepared local accumulation
through raw gradients and three complete updates. No threshold is relaxed.

If prepared local already shows the discrepancy while DDP agrees with it,
the issue is localized before distributed synchronization. That qualifies only
the distributed execution of the prepared path, not independent BF16 numerical
equivalence or harmlessness for training. Fresh-process restart and disposable
capacity checks can then proceed on that path with this limitation visible.
If DDP adds a discrepancy, fix or further localize it before resource checks.

The corrected tiny fresh-process restart pair passes bitwise after GCS restore.
The metadata serialization fix retains weights_only=True; old failure preserved.
