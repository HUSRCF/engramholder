# Follow-up score summaries

These ten snapshots preserve full-precision score aggregates and design
counts from the completed follow-ups. The SHA-256 hashes of each original
`analysis.json` and `records.json` were verified against its completion
record. The lock also records each snapshot's own hash. The snapshots omit
machine-specific metadata, raw predictions, and per-prediction receipts.

Run `python3 scripts/build_rebuttal_assets.py --check` from the repository
root to check the evidence hashes, contrast arithmetic, and generated TeX.
Run it without `--check` to regenerate the tables. The follow-ups reuse
previously evaluated targets and have a separate lock from the original
manuscript results.

The 2026-11-17 extension adds OpenFold MSA32, the retrospective CASP16
subset, and the prospective rotation-prediction test. The original seven
snapshot files and their lock entries are unchanged. Each added source's
analysis and records hashes were checked against its completion record;
the lock also retains its protocol hash. CASP16 has 25 scored targets but
only eight in the primary analysis. Rotation prediction has 192 targets
and eight rotation pairs; its primary intervals resample both units.
The generator preserves CASP16's Bonferroni intervals and the rotation
study's group/target intervals rather than relabeling them as ordinary
target-only intervals. Machine paths and execution-account metadata are
omitted from these anonymous snapshots. The original manuscript v14 lock
is not changed by this separate extension.
