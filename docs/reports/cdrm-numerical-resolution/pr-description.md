# Localize CDRM mixed-precision gradient discrepancies and validate FP32 ordinary attention

At retained update-1000 CDRM weights, the original mixed policy produced
4.44% and 2.57% gradient differences from tiled FP32, with trained Adam-update
differences of 4.63% and 2.21%. The discrepancies persisted with the fabric
bypassed. This change adds reproducible numerical controls, independent
attention/embedding/optimizer references, fixed-cotangent attribution and an
explicit ordinary-attention-FP32 candidate that keeps MLPs, the output head
and the tiled fabric in BF16.

The candidate passes its applicable predeclared screens on both retained
diagnostics and three fresh B64/T256 numerical roles. Fresh trained-case
gradient differences are 0.42% and 0.31%; Adam-update differences are 0.80%
and 0.53%. All unscaled-side and cotangent-scaling checks pass, with exact
observer-removal replays and audited compiled execution. Independent local
attention results equal same-operand FP32 math SDPA followed by the intended
storage cast; independent FP64 Adam reproduces the original trained update
differences, pointing to propagated precision perturbations rather than those
local backward/optimizer implementations.

Retained qualifications: initialization uses the existing cosine gate and
passes at 0.9971, while its first-update relative L2 distance is 7.61%, mostly
from the predefined near-zero-gradient bucket. Strict FP32 raw-side coordinate
flags remain visible, at approximately 5e-7 global relative L2. The overall raw
machine flag and the candidate-specific prospective flag remain distinct.
Failed execution attempts and the initially ineffective ALiBi control are
preserved with their exclusion records.

This milestone changes no original model source, architecture, training
default or numerical tolerance. It performs no training or task-performance
study. The diagnostic wrapper, frozen inputs/sources, CPU and CUDA tests,
W&B plots and verified retained artifacts make the proposed precision
placement reviewable before a separate model-option integration.
