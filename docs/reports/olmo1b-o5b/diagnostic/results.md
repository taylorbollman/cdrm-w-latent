# O5b endpoint: post-hoc feedback diagnostic

Exploratory post-hoc development evaluation of the same trained FBT endpoint; the grid was specified after observing preliminary learning curves and evaluated only after completion. Any best setting is development-set model selection, not a confirmatory result. No weights are trained or changed, no reserved test split is accessed, and RT/NextLat remain off. Beta zero ablates feedback in the FBT-trained backbone; it is not the separately trained ordinary control. Full-window and short-prefix results use different contexts and must not be compared as matched samples. Online evaluation is teacher-forced, not free-running generation. One seed; no efficacy or generalization claim.

Endpoint checkpoint SHA256: `99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66`.

The table scores the final finite pass or exact online state separately; it does not report the summed training CE objective.

| Selection | Beta | Execution | Code NLL | Retention NLL |
| --- | ---: | --- | ---: | ---: |
| 128 windows, max512 | 0 | K=2 | 1.599637 | 3.249787 |
| 128 windows, max512 | 0.25 | K=2 | 1.609754 | 3.284040 |
| 128 windows, max512 | 0.5 | K=2 | 1.622601 | 3.367670 |
| 128 windows, max512 | 0.75 | K=2 | 1.620166 | 3.608247 |
| 128 windows, max512 | 1 | K=2 | 1.634389 | 5.014280 |
| 32 prefixes, max64 | 0.5 | K=2 | 2.180052 | 4.330000 |
| 32 prefixes, max64 | 0.5 | K=3 | 2.179894 | 4.331041 |
| 32 prefixes, max64 | 0.5 | K=4 | 2.180166 | 4.331523 |
| 32 prefixes, max64 | 0.5 | Online | 2.180355 | 4.331842 |
| 32 prefixes, max64 | 1 | K=2 | 2.200133 | 5.083958 |
| 32 prefixes, max64 | 1 | K=3 | 2.200488 | 5.013408 |
| 32 prefixes, max64 | 1 | K=4 | 2.201095 | 5.033363 |
| 32 prefixes, max64 | 1 | Online | 2.200238 | 5.032388 |

All model and buffer hashes unchanged: **True**. No optimizer or checkpoint was created.

Each setting's full per-pass metrics and document records are in [report.json](report.json).
[Standalone figure](diagnostic.pdf) · [PNG](diagnostic.png)
