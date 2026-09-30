# Optional F-only context at update 64

The immutable F-only update-64 evaluation and verified checkpoint publication
are copied to `.runtime/olmo-nfr-kl-continuation/f64-context-01/`. This bundle
does not contain the mutable native-F run report and does not assert that the
ongoing F128 segment has completed.

Use these exact arguments with the completed NFR pair summary, if contextual
comparison remains useful:

```text
--f64-evaluation .runtime/olmo-nfr-kl-continuation/f64-context-01/evaluation-update000064.json 3ebee2842db92e8c532e22fb267ff624f50b6b7da7035d1a832a36038894cef2
--f64-publication .runtime/olmo-nfr-kl-continuation/f64-context-01/publication-update000064.json d25506b4607d1a6fce0448b5b2ef22db2646ee34f2d2b42a8ff894b67090cfd4
```

Both metadata authorities were checked with the summary reducer. They show
active four-pass FBT, beta 1, no RT, no predictor parameters and zero auxiliary
loss weights. `mode.enabled` refers to **FBT**, so it is true for this F-only
model; commit `2ff12d8` corrects the preliminary context guard and adds regression
coverage. The development prefix matches the original NFR parent32 panel, with
65,472 CE targets and common FP32 evaluation without jitter.

| Pass | F64 CE, nats per target |
|---:|---:|
|1|2.675040|
|2|4.536458|
|3|4.688323|
|4|4.765664|

This is descriptive context at the same completed update and on the same data.
It is not a shared-optimizer fork or a KL-only intervention against NFR. Its
different architecture and optimization trajectory prevent that interpretation.
The paired KL comparison still requires completed, audited NFR endpoints; no
NFR summary or W&B publication has been produced by preparing this bundle.

The bundle is retained under
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T070700Z/fbt-stability-f64-context-01/`.
See [storage-receipt.md](../olmo-fbt-stability/storage-receipt.md) for remote
generations and hashes. Publication state/manifest objects were already verified
by the training worker; this preparation did not rehash or upload model state.
