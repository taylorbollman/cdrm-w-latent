# Storage and recovery

All six completed GPU probes, their original205-file source snapshots per case,
and the summary charts/tables are retained with full cloud download-SHA checks.
The small fixture metadata and its289,938-byte tensor payload are separately
retained. Nothing was deleted from the original checkpoints or data.

Cloud prefix:
`gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/20260930T031600Z/`

| Artifact | Child prefix / object | Local receipt |
| --- | --- | --- |
| NF0 forward | `feedback-nf0` | `nf0-01.json` |
| NF32 forward | `feedback-nf32-forward` | `nf32-forward-01.json` |
| NFR32 forward | `feedback-nfr32-forward` | `nfr32-forward-01.json` |
| NF32 gradient primary | `feedback-nf32-gradient-primary` | `nf32-gradient-primary-01.json` |
| NF32 gradient conditional | `feedback-nf32-gradient-conditional` | `nf32-gradient-conditional-01.json` |
| NFR32 gradient primary | `feedback-nfr32-gradient-primary` | `nfr32-gradient-primary-01.json` |
| Summary charts/tables | `feedback-summary` | `summary-01.json` |
| Fixture authorities | `feedback-fixture-metadata` | `fixture-metadata-01.json` |
| Fixture tensor payload | `feedback-fixture/fixture.pt` | `fixture-payload-01.json` |
| Failed first preflight | `feedback-failed-attempts` | `failed-attempts-01.json` |

Receipts live in `.runtime/olmo-feedback-retention/` and contain exact object
generations, SHA256, sizes and verification outcomes. The failed first report is
preserved verbatim under a neutral filename because its obsolete nested source
field does not match the generic evidence archiver's schema. This does not
change its bytes or imply a successful run.

NF0 was restored by the unchanged `scripts.olmo_pilot_execution_restore`, with
generation-bound download and SHA verification. Report:
`.runtime/olmo-feedback-diagnostic/restore-nf0-02/report.json`, SHA256
`05dc4e1e14ffa724e055f5059ee2ba276aa2082c9b624f0adf3467a5a90af247`.
Its SSD location is `/mnt/localssd/cdrm-checkpoints/feedback-diagnostic/nf0-restore-02`.
NF32/NFR32 remain under the existing `adaptation-pilot` SSD root. SSD survival is
not assumed; original checkpoint cloud publications remain the recovery authority.

Final code, test ledger, packaging audit and closeout records are additionally
retained under the closeout prefix recorded in progress.md. No training is active.
