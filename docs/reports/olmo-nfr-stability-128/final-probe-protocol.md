# Saved reduced-KL NFR128 pass curves

After the authorized continuation stops at update 128, observe that saved
checkpoint on exactly the same eight packed T1024 development rows used at 64.
This is an additional weights-only diagnostic, with no optimizer update.

The new `scripts/olmo_nfr_final_curves.py` scope explicitly accepts only the
completed reduced-KL NFR128 continuation. It authenticates the unchanged
continuation resolution, source inventory, terminal report and unique verified
publication; the update64 observer and all historical training sources remain
unchanged. Bind the final report/publication SHA only after training completes,
cloud128 is verified and W&B is synchronized. Release training's GPUs before
running the observer on one explicitly assigned GPU in the required container.

Reuse the update64 observer's canonical K32 forward and every measurement
definition: FP32, no jitter, no predictor calls, consecutive changes, regional
CE/entropy/scales and direct K4/K8 hidden residuals against finite K32. Retain
the predictor's saved tensors, preserve model/runtime/RNG/input state, and
compare the exact input tensor hashes against the completed reduced64 probe.
Use the same W&B account and retain the small report and figure artifacts.

K32 is a finite reference, not exact online. These curves assess whether the
four-pass truncation discrepancy changes during the continued optimization;
they do not establish quality, global contraction or RT's independent value.
Regular 64-row development evaluations at 96, 100 and 128 remain separate from
this eight-row diagnostic. Additional deep probes at 96/100 are not planned
unless an observed health concern justifies them.

Preparation can run on CPU while training continues. The final scope is created
with `make_scope(references, checkpoint)` only after immutable terminal pins
exist. Its references are the completed reduced64 observer, the continuation
scope and dry resolution, the completed128 training report and its128 cloud
publication. Then run the helper with `--preflight-only` in the CPU container.
After that passes and the assigned GPU is idle, root may run the same command
without that flag into a fresh directory. Prior matched probes took about
8.5 minutes per checkpoint; this is a rough runtime estimate, not a guarantee.
