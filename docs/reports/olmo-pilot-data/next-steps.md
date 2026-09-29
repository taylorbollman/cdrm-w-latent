# Next milestone: connect the ordered corpus to the accepted runner

The data preparation milestone changes neither the model nor its numerical
qualifications. In particular, recoverable data and correct loss masks do not
resolve BF16-versus-FP32 trajectory differences. Do not reopen that study or
launch a quality campaign automatically.

The existing shared SSD engine accepts a reader and prebuilt plans. Its batch
materialization and cursor checks can use the new `OrderedCampaignData` API.
Keep that engine, model code, old canonical-order reader and historical execution
contracts frozen. Add three versioned adapters:

1. `olmo_pilot_execution_contract.py`: authenticate the corpus, suite/catalog,
   selected train index, recipe/order/exclusions and acquisition mapping; resolve
   the finite metadata plan and resource allocation. Use a new ordered-data
   execution identity and complete new source inventory. Never relabel an old
   packed checkpoint or silently migrate its cursor into this stream.
2. `olmo_pilot_eval_control.py`: resolve named dev panels with the existing
   per-pass numerical evaluation and live-state preservation helpers. Declare
   evaluation physical batch independently if common-FP32 memory requires it.
   Keep denominators separate by CE, latent pairs and KL triples. Main and source
   panels overlap, so never pool them as independent observations.
3. `olmo_pilot_execute.py`: resolve before CUDA, create the ordered reader and
   connect accepted model construction/startup imports to the unchanged shared
   SSD engine, checkpoint retention and recovery primitives.

Prepared confirmation panels are not scheduled development evaluations. They
need a later explicit confirmation invocation after the comparison decision.
The 21 prepared panels do not imply evaluating all of them at every checkpoint.
Estimate actual common-FP32 evaluation cost before choosing the small routine
monitoring panel and cadence; freeze that membership before learning.

Acceptance starts with CPU plan identity and literal data/count/partition checks,
then a bounded tiny two-rank graph case with a cross-document row and an uneven
last slot. Insert dev evaluation and verify that parameters, Adam, gradients,
RNG, training cursor, graph buffers and precision/modes are preserved. Stop at a
completed boundary, retain to GCS, restore in a fresh process and require the
next update and complete state to match. Reject changed round/order, membership,
source authority or execution partition on resume.

After that integration passes, measure native ordinary B and combined NFR at
T1024 on the current two H100s. Use BF16, graphs, accepted fusions and activation
checkpointing with no new LR or precision sweep. Start NFR near B12/rank, retain
B8 as a fallback, and measure actual physical batching rather than attributing
large-batch RT utilization to accumulation. Report valid-input throughput and
memory separately from graph setup, common-FP32 evaluation and checkpoint
writes/transfers. H100 measurements are not H200 capacity acceptance.

Startup remains an explicit decision. Original weights with fresh Adam are the
common starting point; the accepted alternative imports only fusion weights
from 128 FP32 fusion-only steps for NF/NFR and starts fresh all-active Adam.
That adaptation saw 1,073,565 valid inputs and 1,048,576 CE targets; it does not
import the adapted full-NFR backbone, predictor or optimizer. Capacity fixtures
may use these routes, but they are not a matched quality cohort. Freeze startup,
exposure, arms, finite token budget, schedule, monitoring and storage capacity
before a learning pilot. The 134M-token data panel is capacity, not that decision.
