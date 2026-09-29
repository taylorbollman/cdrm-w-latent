# Development prefix coverage before selecting monitoring

The fixed PR49 `dev-main` prefixes have the following actual metadata coverage.
This inspection used the pinned ordered reader and original document intervals.
It did not evaluate a model, reorder data or select the final monitoring panel.

| Valid input tokens | T1024 chunks | Unique documents touched | Source objects touched | Represented strata | Missing strata |
| ---: | ---: | ---: | ---: | ---: | --- |
| 5,120 | 5 | 14 | 12 | 4 of 9 | Books, C4, peS2o, Reddit, Wikipedia |
| 16,384 | 16 | 52 | 22 | 6 of 9 | Books, peS2o, Wikipedia |
| 65,536 | 64 | 193 | 29 | 7 of 9 | Books, Wikipedia |

The current 5,120-input capacity prefix contains only the three Common Crawl
strata and Stack. It is suitable for exercising execution and estimating small
evaluation cost, but should not be described as covering the entire proposed
mixture. The larger prefixes add coverage without changing the existing order;
even 65,536 inputs do not include the very low-weight books or Wikipedia strata.
Nine strata are fewer than nine independent source families: the three Common
Crawl strata belong to the same family.

Each cell below is **chunks / unique documents touched**. Each full chunk
contains 1,024 valid inputs. Source-object counts above refer to the acquired
objects, not source families or distinct websites.

| Stratum | 5,120 inputs | 16,384 inputs | 65,536 inputs |
| --- | ---: | ---: | ---: |
| Books | 0 / 0 | 0 / 0 | 0 / 0 |
| C4 | 0 / 0 | 1 / 2 | 4 / 14 |
| Common Crawl head | 1 / 3 | 3 / 10 | 10 / 24 |
| Common Crawl middle | 1 / 1 | 3 / 4 | 13 / 30 |
| Common Crawl tail | 2 / 7 | 6 / 25 | 25 / 98 |
| peS2o | 0 / 0 | 0 / 0 | 2 / 2 |
| Reddit | 0 / 0 | 1 / 7 | 2 / 12 |
| Stack | 1 / 3 | 2 / 4 | 8 / 13 |
| Wikipedia | 0 / 0 | 0 / 0 | 0 / 0 |

These are nested prefixes. The 5-chunk prefix is entirely contained in the
16-chunk prefix, which is entirely contained in the 64-chunk prefix. Their
documents and outcomes would therefore overlap; they are not independent
evaluation sets. Unique-document counts use the original corpus document index,
cross-checked against document key, source index/name/line and ID. Multiple
fragments from the same document count once within a prefix. For example, the
64-chunk prefix contains 198 document segments but touches 193 unique documents.
These counts do not imply evaluating each complete document. Embedded EOS
tokens do not create artificial document identities.

| Inputs | CE targets | NextLat pairs | KL triples | Cross-document CE targets |
| ---: | ---: | ---: | ---: | ---: |
| 5,120 | 5,115 | 5,106 | 5,092 | 9 |
| 16,384 | 16,368 | 16,332 | 16,280 | 36 |
| 65,536 | 65,472 | 65,338 | 65,140 | 134 |

The later monitoring choice should combine these coverage facts with measured
evaluation cost. A larger main prefix and separately reported source panels are
available options; this note chooses neither. Main/source overlap and the small
book pool remain subject to the [full corpus assessment](../olmo-pilot-data/coverage-assessment.md).
No confirmation panel was opened for this inspection, and no model outcomes
were observed.

Evidence is `.runtime/olmo-pilot-execution/dev-prefix-coverage-01/report.json`,
SHA256 `ec913e41bb5a0a773c4ac98aac6297f8895022e30210b7d139b039fe7f7fbfd2`.
It retains the helper, accepted reader source pins, reference report,
declaration, suite/dev manifests, acquisition mapping and source-authority pins.
The dev SQLite size/hash is recorded without copying the large index. The probe
completed in approximately 1.07 seconds in a GPU-disabled project container.
Corpus bytes were authenticated, but token payloads were not materialized and
the reader cursor did not advance. No GPU, model loading or network was used.
