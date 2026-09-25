# Bounded fixed-state follow-up

Frozen after `actual-eager-01`, before the follow-up GPU run (2026-09-25).
The original report remains **failed**; no tolerance is changed.

Ordinary and RT both pass two independent-trajectory updates. Combined passes
update1 (raw-gradient relative L2 2.343e-9), then update2 has raw-gradient L2
1.350976e-5, still inside the original gradient budgets, exact rank replicas and
matching loss/count metrics. However, 20 parameter tensors and7 Adam moment
tensors miss the stricter elementwise complete-update tolerances. Maximum
parameter absolute difference is1.383945e-6; maximum parameter tensor relative
L2 is9.0963e-8. The original failed result is retained in GCS with the successful
earlier cases and complete per-tensor diagnostics.

Hypothesis: the extremely small reduction-order differences after the first
update can change BF16 rounding decisions during the next forward/backward;
Adam can amplify differences in small/cancelling gradient coordinates. This is
an explanation to test, not an established excuse or numerical clearance.

Follow-up: combined only, same B1/rank, T512, two accumulation microbatches,
two optimizer updates, same native checkpoint/runtime and unchanged budgets.
Each rank independently computes the canonical rank-major reference trajectory.
After recording each candidate update, restore canonical model/Adam/scheduler/
counters in place before comparing the next step. This checks the distributed
update at an identical starting state. It deliberately does not establish
agreement between independently evolving BF16 trajectories.

If anchored checks pass and replica/recovery checks remain exact, proceed with
the functionality/efficiency milestone while retaining the independent-trajectory
qualification. If they fail, localize the fixed-state reduction/update discrepancy
before accepting the affected execution path. Do not broaden into a quality or
long-run precision campaign without a concrete reason.
