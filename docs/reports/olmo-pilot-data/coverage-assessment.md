# Actual pilot corpus coverage

The completed corpus meets the declared nine-stratum, T1024 panel quotas. It is a bounded sample for the next training-readiness milestone, conditional on the selected object prefixes. It does not establish population representativeness, model quality or executor compatibility.

The retained candidates contain **584,851 unique documents and 310,669,141 tokens** from all 37 selected source objects. Acquisition supplied 584,894 complete raw records and 310,670,964 candidate tokens; exact-content deduplication removed 43 records and 1,823 tokens. All 592 independently selected audit documents—16 per object—retokenized exactly without truncation. Full shard integrity and all document metadata were checked; independent native retokenization covered those 592 documents, not every candidate.

The three main panels have these actual sizes. “Selected documents” counts unique documents touched by a selected chunk, including partial documents; it is not the number of document completions.

| Panel | Full chunks | Input tokens | CE targets | Reserved documents | Selected documents |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 131,072 | 134,217,728 | 134,086,656 | 238,913 | 238,861 |
| Development main | 1,024 | 1,048,576 | 1,047,552 | 3,113 | 2,095 |
| Confirmation main | 1,024 | 1,048,576 | 1,047,552 | 2,884 | 1,922 |

All panels have zero padding. Each of the 18 additional source panels contains 64 chunks, 65,536 inputs and 65,472 CE targets. The train stream contains 238,608 cross-document CE targets; these remain supervised. Its NextLat counts are 133,848,048 latent pairs and 133,478,846 KL triples, excluding the applicable true-document boundaries. Chunk boundaries do not predict into another chunk, and an embedded EOS token alone does not create a document boundary.

## Quotas and actual training coverage

The last column applies separately to both main held-out panels. Chunk quotas are exact; source-object counts describe acquisition, not the final token mixture.

| Stratum | Acquired objects | Train chunks | Train reserved documents | Train selected documents | Main held-out chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Books | 3 | 255 | 3 | 3 | 2 |
| C4 | 4 | 8,549 | 18,231 | 18,231 | 67 |
| Common Crawl head | 4 | 20,930 | 29,336 | 29,336 | 163 |
| Common Crawl middle | 4 | 26,480 | 36,923 | 36,920 | 207 |
| Common Crawl tail | 8 | 51,144 | 112,661 | 112,657 | 400 |
| peS2o | 4 | 2,800 | 5,706 | 5,706 | 22 |
| Reddit | 4 | 3,930 | 18,513 | 18,513 | 31 |
| Stack | 4 | 16,802 | 17,254 | 17,209 | 131 |
| Wikipedia | 2 | 182 | 286 | 286 | 1 |

Training touches 36 of the 37 acquired objects; `books-01` contributes no selected training tokens. Both main held-out panels touch 35 objects. The nine source panels collectively touch all 37 objects within each held-out split. All nine strata are represented in every main panel.

## Held-out reserves versus evaluated coverage

Whole-document reservation ensures enough tokens and at least eight book documents or 32 documents for each other stratum. The later chunk selection can touch fewer documents. The source-panel column below counts actual selected documents in its 64 chunks, not the reserve minimum.

| Stratum | Dev reserved | Dev main selected | Dev source selected | Confirmation reserved | Confirmation main selected | Confirmation source selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Books | 8 | 2 | 8 | 8 | 2 | 5 |
| C4 | 295 | 186 | 180 | 281 | 167 | 159 |
| Common Crawl head | 229 | 229 | 108 | 173 | 170 | 101 |
| Common Crawl middle | 304 | 304 | 129 | 261 | 261 | 111 |
| Common Crawl tail | 988 | 988 | 243 | 958 | 958 | 214 |
| peS2o | 231 | 59 | 135 | 277 | 49 | 137 |
| Reddit | 616 | 162 | 330 | 620 | 176 | 351 |
| Stack | 164 | 164 | 98 | 151 | 136 | 90 |
| Wikipedia | 278 | 1 | 167 | 155 | 3 | 97 |

Books are the clearest coverage limit: only eight eligible development books were available, exactly meeting the reservation minimum. Eleven eligible confirmation books were available after exclusions; eight were reserved and five appear in the confirmation source panel. Training uses three books, while each main held-out panel has only two book chunks. Wikipedia has only one chunk in each main panel. Source-panel results will therefore be useful alongside the weighted main metric, but the book results will remain based on very few documents.

Within each held-out split, the main panel overlaps the source panels by **376 exact chunks**. Their union contains 1,224 distinct chunks: 2,511 development documents or 2,282 confirmation documents. These panels are complementary views of shared data, not independent replications. Repeated chunks must not be counted twice when describing unique evaluation exposure.

## Separation, selection and limits

The fixed content-token hash rule with seed 20260929 assigns whole documents to train/dev/confirmation with 90/5/5 probabilities. It produces 526,578 / 29,088 / 29,185 candidate documents respectively; it does not enforce exact token percentages. Selection uses the frozen document and chunk hash orders and integer quotas, with no model outcomes involved.

The 12,512 prior-readiness content identities are excluded from pilot selection. The candidate storage still contains **2,119** such identities, covering 2,046,551 tokens: 14 books, 2 Common Crawl tail records and 2,103 Wikipedia records. Keeping them in candidate storage is intentional; no reserved or selected panel document is marked excluded. Exact content-hash intersections between train, development and confirmation are zero. This excludes exact token-content reuse, not near duplicates or unknown exposure during OLMo pretraining.

The source-family weights are the declared provisional mixture. Common Crawl subweights use object counts as a proxy for token mass. Hash-selected objects followed by bounded complete-record prefixes do not form a uniform sample of all Dolma documents, nor reconstruct OLMo's historical training stream. The immutable training panel supplies 134,217,728 unique input-token positions; a larger training exposure requires explicitly authorized new rounds or reuse. Data availability does not authorize a training budget.

## Assessment evidence

This assessment rechecked all 21 panel manifest pins, source/recipe/corpus agreement, raw-audit versus suite availability, and SQL reserved/selected document counts and source-token totals. It independently recomputed the content-hash split for all 584,851 candidates, checked selected membership separation and exclusion flags, and counted actual same-split chunk overlap.

The separate ordered audit on the recovered corpus and suite also passed. It counted intervals and per-loss metadata for all 134,272 panel chunk entries, reconstructed 336 sampled chunks literally (16 per panel), and checked rank masks and cursor behavior in all 21 panels. Its selected-document and overlap totals agree with this assessment. Its token reconstruction is sampled; it does not claim independent retokenization of all stored tokens.

| Authority | SHA256 |
| --- | --- |
| Raw/token audit report | `b6053115ea9cf5101fc0cd29cd82f56af16bf6644dc64bd103c77b2505902484` |
| Ordered build report | `7b5275fa4d28b2513fbe4ce5b35872c1c4d89239f0d981e1814b664f03609101` |
| Recovered ordered-data audit | `414fd5ab28b852757eb63e973ed24757a8b8e50360c602ce858c7102673db990` |
| Candidate corpus manifest | `f3206c360dd412ccbaf08c65b30a7ba7f1b30b56db7ffbb9559522831c797e06` |
| Ordered suite manifest | `080402225e24350cc377b720bdf39a55b87e71e4a6af5fea3cda7460eb169215` |
| Canonical recipe | `d826a3d005c3dfa56b8f1dbe882cd91f84d0d41922cac97563ec640cb5f93ba6` |

The reports are under `.runtime/olmo-pilot-data/{raw-token-audit-01,ordered-build-01,ordered-audit-01}/report.json`; the suite is `/mnt/localssd/cdrm-data/olmo-dolma-v1_5-pilot-20260929/ordered`. This review involved no model loading, GPU execution or new corpus acquisition.
