# Project instruction provenance

These files are byte-for-byte copies of the user-supplied documents in
`/home/taylorbollman/`, captured on 2026-09-06:

- `recurrent_transformer_coding_agent_brief.md`: original architecture definitions
  and implementation instructions.
- `coding_agent_review_amendments.md`: advisory correctness and staging review.
- `r3_bf16_mixed_precision_agent_brief.md`: BF16 autocast implementation,
  diagnosis, bounded training/recovery, and performance guidance.
- `r3_backward_validation_request.md`: bounded investigation of the remaining
  FP32 recurrent-backward uncertainty, authorized after Stage B.
- `recurrent_transformer_initial_run_protocol.md`: preliminary, provisional run
  protocol; its schedules, budgets, metrics, and checkpoint names are proposals.

The user explicitly made the review and protocol advisory and delegated technical
judgment to the coding lead. After completing Stage A numerical/operational
groundwork, the user authorized the next milestone: the small Stage B synthetic
pilot. Its resolved configuration and bounded execution are recorded in
[`docs/stage-b-plan.md`](../stage-b-plan.md). Proposed later research budgets and
checkpoint names are not evidence of authorization, execution, or artifact
existence. Actual runs and results are recorded separately under `docs/reports/`.

Source hashes are recorded in `source_sha256.txt`. Input copies retain their
original contents; later project decisions belong in separate records.
