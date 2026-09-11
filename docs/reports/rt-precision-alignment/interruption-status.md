# Status after interruption, 2026-09-10

The protected first-seed continuation stopped after recorded update 169. Its
last complete JSONL record is 168; the final JSONL row contains zero bytes.
The atomic progress record includes finite update 169, but no model/Adam
checkpoint was saved after 100. Preserve the interrupted directory unchanged.
No training process is currently active; the required container can access
the H100 80GB.

Both first-seed 100-update runs, development evaluations, physical B512 trained-state
A/B/C comparisons and the original fresh-process resume proof completed.
Both 100-update checkpoint SHA256s and all training source hashes were reverified
after interruption. Data/checkpoints and both trained numerical packets have
verified GCS retention. The local compiler cache remains present; its complete
integrity has not been reaudited after interruption.

Restart from the original A seed0 step 100 checkpoint in a fresh output
directory, after a brief runtime/resume check. Replaying 101–169 costs roughly
seven minutes of training plus setup. Then complete A/B seed0 to 500, the
preselected second pair to 500, and the planned endpoint assessments.
No 500-update endpoint or confirmation evaluation has completed.

The [100-update report](pilot-100-results.md) records the current evidence:
trained numerical screens passed, but B had a +0.009403-nat/token development
deficit at 100. Keep A as default and B experimental while completing the
frozen comparison; retain the 0.005 margin and full 5000-update warmup.

Execution lineage: `.runtime/rt-precision-alignment/20260910T191100Z`.
