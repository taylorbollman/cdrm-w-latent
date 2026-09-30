# Progress

2026-09-30 15:24 UTC: began the user-authorized endpoint probes and conditional
KL0.1 NFR64→128 milestone. Both H100s verified idle in the required project
container; no prior queue/training process remains. The completed NFR64 pair,
its populated checkpoints and cloud receipts are available. SSD has about
1 TiB free; persistent boot disk about 88 GiB. Large new states remain on SSD
with verified cloud retention.

Branch: `feat/olmo-nfr-stability-128`, from main `4b15854`. Implementation is
being prepared in separate, explicit diagnostic and continuation scopes.
See [protocol.md](protocol.md). GPU execution has not yet started.

15:39 UTC: both saved-state endpoint probes launched in isolated one-GPU
containers after 44 focused CPU tests, authenticated real checkpoint preflights
and independent review. Control uses physical GPU0, reduced uses GPU1.
Host sessions 2942/66539; read-only monitor 25914. Both are online:
[KL1](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/y8xlyhtw),
[KL0.1](https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/sf5dwyaz).
Reports are `.runtime/olmo-nfr-endpoint-curves/result-{control,reduced}-01/report.json`;
launch receipts/logs are adjacent `launch-{control,reduced}-01` directories.
Scope SHA `e8d52df40aa5456639c919ff32d58cde12a65a3fcf49df07d91778f218d48880`;
launch plan SHA `a4001afe27885330c777db707e0eae6884472730e7271a0ddf4c4685efc05cee`.
Each has a 30-minute timeout. No training is running beside them.

The continuation passed 36 focused CPU checks and authentic metadata dry
resolution. Its scope is
`.runtime/olmo-nfr-stability-128/continuation-prepared-01/scope.json`, SHA
`563c6bbb3b5491bffae75879fbfc9299b94021a25c8c391a91d451a1622d4523`.
The 222-source inventory and protocol are frozen. The adapter preserves the
ten accepted model/update/save/graph callbacks, strictly loads old64 state,
then records only the new execution/observation authority. Activation awaits
the completed probes; no further user permission is required for the agreed
conditional continuation. PR56 is a draft; code and handoff are being saved.

15:48 UTC: both probes completed all eight rows, preservation passed, W&B
synced and host launchers exited0. Summary3p91owsx is published; plot inspected.
Control report SHA `6a80643b3855829e4ad444a0a42dc188181348e7e9899879c40fe1d4ae276ac9`;
reduced `97fe4e9029f809c7bc46956b8fee989e590a6f9839959e8bef64f72cd4378a0b`.
Both show bounded/decaying changes. Reduced K4/K8 tail residuals to finite32
are2.1792%/.1766%; more passes slightly worsen CE5.412258→5.431623. See results.

15:49 UTC: activated unchanged reduced64→128 after that assessment. Both
GPUs verified released inside the new training container before launch.
Hostsession83327; read-only monitor68687. W&B
https://wandb.ai/taylorbollman/pretrained-fbt-rt-nextlat/runs/2zu5jloq .
Activation receipt `.runtime/olmo-nfr-stability-128/activation-01.json`, SHA
`9a85cd4f6b2d48e95a0e301c92f6536bee302acb3b5af0496da15c1247c95f71`.
Run report is `.runtime/olmo-nfr-stability-128/native-nfr-reduced-64to128-01/report.json`;
host log/launch receipt are adjacent `native-nfr-reduced-64to128-01.log` and
`native-nfr-reduced-64to128-01-launch.json`. SSD root is
`/mnt/localssd/cdrm-checkpoints/nfr-stability-128/native-nfr-reduced-64to128-01`.
The external timeout is6hours; STOP.json in the runtime root requests a
checkpointed stop. Do not start another GPU job while this two-rank run lives.

After terminal128, the final saved state will receive the same one-GPU K32
observation, using a new explicit authority outside the live training pins.
No additional intermediate deep-pass grid is planned absent a specific concern.

16:07 UTC: strict restore, metadata transition, repeated dev64 and graph
preparation all preserve the complete saved boundary exactly on both ranks.
Repeated dev64 raw rows/counts/sums and per-pass CE/latent/KL match prior64.
Immutable origin/evaluation evidence is retained; see validation/storage notes.
Update65 completed with finite metrics, preclip norm2.544913 and LRused.0001352
(next.000137). First new checkpoint65 save/publication is underway. Training
continues unchanged; next regular dev80.

Final128 observer is prepared with30focused CPU tests and independent review:
`.runtime/olmo-nfr-final-curves/preparation.json`, SHA
`e00655584ae906401bc12bab826ae24c68edf830a75819bae7327c8e13dddde8`.
After terminal128/cloud128/synced/hostexit0, run the guarded CPU audit:
`python3 .runtime/olmo-nfr-stability-128/run_final_audit.py`.
Then bind/preflight final curve inside CPU-only container:
`python .runtime/olmo-nfr-final-curves/prepare.py --preparation-sha256 e00655584ae906401bc12bab826ae24c68edf830a75819bae7327c8e13dddde8 --terminal-report-sha256 VERIFIED_FINAL_SHA --publication-sha256 VERIFIED_PUBLICATION_SHA --output-dir .runtime/olmo-nfr-final-curves/bound-01`.
It produces the explicit single-GPU launch command. Check GPU release before
launching. New probe/summary code is outside frozen222 training pins. Use
`scripts.olmo_nfr_128_summary` for audit-bound training plots and
`scripts.olmo_nfr_endpoint_summary --final128 FINAL_REPORT` for reduced64/128
pass overlays; preserve original paired64 artifacts separately.

16:20 UTC: updates65–70 all finite with identical reduced metrics across ranks;
raw norm min/median/max2.545/3.129/3.915, clipping remains active. Cloud65
verified; local70 save is underway. Graph memory is58.38GiB reserved and
42.83GiB peak allocated per GPU, with13.46GiB sampled free after capture.
No setting or code change. Dev80 remains the next predictive readout.

16:52 UTC: dev80 complete, state preservation exact on both ranks. CE passes1–4
is2.787306/4.407264/4.511726/4.563218, versus64
2.774970/5.375003/5.481358/5.520121. First pass rises.012336nats; fourth falls
.956904, reducing the fourth-minus-first gap2.745152→1.775912. This is
substantial feedback adaptation but not useful refinement. Raw latent is
.225323/.116845/.113217/.112122 and KL3.069052/1.618650/1.621195/1.613884:
later-pass auxiliaries increase while CE improves. All16 resumed updates
finite/clipped; norm min/median/max2.359/3.129/5.199. All222live source pins
remain exact. Cloud75 verified; named80 save/publication follows evaluation.
Next regular dev96, then100 at the existing warmup boundary; continue unchanged.

17:14 UTC: update88 complete; all24 resumed updates finite/clipped, identical
rank metrics. Norm min/median/max1.585/3.033/5.199; latestLR.0001766. Cloud85
verified. No new development measurement since80 and no setting changes.
Guarded terminal W&B checkpoint-summary reconciliation/retention is prepared
in `.runtime/olmo-nfr-stability-128/retain_terminal_128.py` (no active-run
mutation); see storage-receipt.md for the pinned command and recovery notes.

17:40 UTC: dev96 complete and boundary-exact on both ranks. CE passes1–4
is2.774441/3.630273/3.676090/3.700264. Fourth-minus-first gap is.925822,
down from1.775912 at80; useful refinement remains unestablished. Raw latent
is.205328/.138491/.138226/.137641; KL2.867770/1.967461/1.977498/1.971348.
All32 resumed updates finite/clipped; norm min/median/max1.585/2.744/5.199.
Cloud95 verified; named96 save follows evaluation. Continue unchanged through
the existing warmup100 boundary to128. No other GPU work has been launched.

17:55 UTC: dev100 complete/boundary-exact. CE passes1–4 is
2.769629/3.510893/3.552609/3.574547, gap.804918. Raw latent
.205370/.144993/.145093/.144664; KL2.847800/2.045665/2.048391/2.041702.
All36 resumed updates finite/clipped; norm min/median/max1.472/2.580/5.199.
Update100 uses.0001982 and schedules.0002 for101; this is the original
warmup, not a changed learning-rate plan. Cloud96 verified. Remaining28
updates continue unchanged at the planned peak LR; next regular dev112.
