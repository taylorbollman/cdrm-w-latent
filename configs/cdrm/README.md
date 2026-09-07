The active configuration is **six ordinary blocks, D128/H16/MLP512**, with a
CDRM side scan after block 3 sharing ordinary block 1. The completed SEQ6 screens
selected `copy-v16-t256-k96` for separate seed-0 and seed-1 CDRM comparisons at
25 common epochs. The [frozen selection](../../docs/reports/cdrm-naive/six-comparison-selection.json)
records that development-only choice. Recall V128 is deferred. The [runtime-only continuation policy](../../docs/reports/cdrm-naive/six-continuation-policy-update.json)
chooses the first fitting common endpoint 50/45/25 from measured wall durations
after all four epoch-25 runs complete, without accuracy-dependent allocation.

The base files are authoritative inputs to `scripts/cdrm_common.py`; topology
files are complete `ModelConfig` snapshots for inspection or direct Python
construction. The CLI uses `--preset` and `--topology`, without a `--config`
option. Base files were not changed to create these snapshots. The original
[base provenance](provenance.json) identifies the retained saved configuration
and the deliberate width adaptation.

| Profile prefix | Backbone | Actual data shape | Status |
|---|---|---|---|
| `d128_6_copy_v16_t256_k96_*` | 6 blocks, D128, H16, MLP512 | Official MAD copying T256, V16, K96 | Selected seed-0/1 comparison |
| `d128_6_recall_v128_t128_*` | 6 blocks, D128, H16, MLP512 | Official MAD recall configured T128, actual T127, V128 | SEQ6 screening complete; CDRM deferred |
| `d128_mad_recall_*` | 12 blocks, D128, H16, MLP512 | Original MAD recall actual T127, V16 | Historical SEQ12 E50 calibration; CDRM12 SYN never launched |
| `d256_mqar_*` | 12 blocks, D256, H4, MLP1024 | Existing custom Stage B MQAR T128, V1024 | Separate NUM/benchmark profile; CDRM training integration unsupported |

The [six-block manifest](joint_d128_6_screening_first.json) contains both harder
settings, their pinned official difficulty provenance, resolved profile/data
hashes, and explicit screening/comparison command templates. The old
[D128 joint plan](joint_d128_mad_recall.json) is marked superseded before paired
research execution. The [D256 manifest](joint_d256_mqar.json) remains prospective
for research and provides only supported NUM/benchmark commands.

| Suffix | Six-block topology |
|---|---|
| `seq` | All six ordinary blocks; side memory disabled. |
| `cdrm` | Ordinary preview through block 3, one side scan sharing block 1, bridge into block 4. |
| `same_depth` | The same scan/bridge, with an active shallow candidate adapter input; deferred attribution arm. |
| `r3` | Existing rho-1 recurrent replacement at block 1; no CDRM side memory; deferred attribution arm. |

Sites use zero-based indices. Let `p_early` and `p_late` be outputs after ordinary
blocks 1 and 3. The deep candidate is
`p_early + 0.1 Ad(Nd(p_late-p_early))`; active same-depth uses
`p_early + 0.1 Ad(Nd(p_early))`. Both retain the ordinary late preview and bridge
through `p_late + 0.01 Ab(Nb(hat_m-p_early))` into block 4. Blocks 4–5 form the
ordinary suffix. Rho is 1, so persistent records project the proposed state.
Adapters are separate, nonzero, bias-free D-to-D matrices initialized with normal
standard deviation `1/sqrt(D)`. Stateless RMS normalizers use epsilon `1e-6`.

The active same-depth control does **not** replace `p_late` with `p_early`, which
would zero the deep difference input and is a separate removal ablation.
Different adapter-input statistics remain a limitation of the active control.
Rho 0 is not a SEQ switch. Lambda 0 is the matched numerical SEQ bypass.
`cdrm_read_mode="current_only"` keeps only the current temporary K/V pair in a
read; published profiles use history. Compatibility diagnostic keys `p3`, `p8`
and `v8` denote the configured early preview, late preview and bridge. Their
six-block sites are 1/3/4; the historical twelve-block sites are 3/8/9.

All profiles retain pre-norm learned LayerNorm, learned QK norms, GELU, ALiBi
maximum 8, no biases/dropout/embedding norm, an untied head, and Mitchell backbone
initialization. `max_sequence_length=512` is attention-bias capacity; actual
inputs are not padded to 512. Execution is FP32 with autocast and TF32 disabled.
R3 uses its existing compiled tiled backend with four MLP chunks; CDRM uses
ordinary autograd without a custom backward.

Physical batch is 128, without accumulation. AdamW starts at `5e-4`, with betas
`(0.9,0.98)`, epsilon `1e-8`, weight decay 0 and clip norm 1. Cosine scheduling
keeps its 200-epoch horizon, steps after completed epochs, has minimum LR `1e-6`
and no warmup. Screening stops do not shorten the schedule. Seed 0 and seed 1
remain separate matched comparisons; shuffle seed 45678 preserves the retained
permutations. Final data is reserved until selection is frozen, and evaluations
require explicit checkpoint/split commands. Manifests describe configurations
and intended commands; run reports determine which experiments completed.

The [usage guide](../../docs/cdrm-naive-usage.md) covers container entry, screening,
paired fresh initialization, continuation, explicit final evaluation and CPU
report generation. Native recall training CE and answer-only development CE
measure different targets; their difference is not an ordinary generalization gap.
