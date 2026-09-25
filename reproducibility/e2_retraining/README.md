# Same-seed repetition of nine compensated Factor fits

All nine original `+C` conditions were retrained for 1,536 steps and evaluated
on the original 96 targets. The no-C baseline, training seeds and selected
rotations are reused. These are not additional independent seeds or targets;
the original and repeated executions are kept separate.

```bash
python scripts/verify_e2_retraining.py
```

The verifier starts from the new 864 score records plus the original no-C
records. It reconstructs the 21 contrasts, their target-bootstrap intervals
and seed marginals, checks the unchanged baselines, and checks provenance
against the archived protocol, code, per-fit receipts, engineering gate and
CIF audit. It does not rerun training or independently parse the CIFs.

The archived execution completed the six-condition 384-update training/resume
check and the 13,824 formal updates. Its CIF round-trip audit found identical
masked coordinates and scores. The repeated rotated-arm benefit was positive,
but additional benefit over Native was not re-established. Neither execution
is discarded, pooled or selected as the preferred result.

The source snapshots retain their original dependency and cluster-path
requirements. This repeat reused the verified environment and ESM2 caches;
it is not the separate fresh-FASTA/feature replay, a clean installation or a
fully portable retraining release. The source of training divergence remains
unidentified; engineering restoration checks do not establish strict FP32
trajectory equivalence.
