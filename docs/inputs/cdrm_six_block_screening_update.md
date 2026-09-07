# User update: six-block development screening

Received in the project conversation on 2026-09-07, during initial pilot execution.
This supersedes the earlier twelve-block paired-pilot scheduling instructions.

Screen a small set of harder established synthetic task settings with SEQ-6 first.
Use development learning curves, scored-token accuracy and whole-sequence exact
match to decide where to invest CDRM training. A setting that SEQ-6 reaches
near-perfect performance on quickly and reliably is low priority: skip CDRM there
and move to a harder setting. Substantial learning with remaining errors is a
priority comparison. Near-chance or shortcut-level performance does not exclude
CDRM, but first check optimization and supervision. The hardest task need not be
the most informative. Solving a task only after a long run can still leave a
learning-efficiency question, behind settings with a clear accuracy gap.

Freeze the selected setting and compare on fresh held-out data. Retain SEQ's
initialization, data order, configuration and checkpoints so a compatible SEQ
baseline can be reused when CDRM trains. Development screening allocates training;
the final test assesses the selected comparison.

Implementation choice pending an optional user preference: six-block early/late
sites1/3, leaving ordinary suffix blocks4/5. Both1/3 and1/4 pass targeted CPU
composition tests. This site choice is not based on synthetic outcomes.
