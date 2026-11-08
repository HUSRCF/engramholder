# Follow-up score summaries

These seven snapshots preserve full-precision score aggregates and design
counts from the completed follow-ups. The SHA-256 hashes of each original
`analysis.json` and `records.json` were verified against its completion
record. The lock also records each snapshot's own hash. The snapshots omit
machine-specific metadata, raw predictions, and per-prediction receipts.

Run `python3 scripts/build_rebuttal_assets.py --check` from the repository
root to check the evidence hashes, contrast arithmetic, and generated TeX.
Run it without `--check` to regenerate the tables. The follow-ups reuse
previously evaluated targets and have a separate lock from the original
manuscript results.
