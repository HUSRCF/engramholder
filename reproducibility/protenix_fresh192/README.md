# Fixed-model Protenix Fresh192 validation

This directory preserves the completed study's protocol, interpretation,
per-target CSV and eight execution-locked source snapshots. The snapshots
document the original implementation; their cluster paths are not a portable
inference environment. The score arrays and model/target/selection/execution
manifests are under `evidence/protenix_fresh192_*.json`.

Run from the artifact root:

```bash
python scripts/verify_protenix_fresh192.py
python tests/test_protenix_fresh192.py
```

The verifier checks the complete 25-model-by-192-target grid, raw score
pairing, all 24 metric/contrast summaries and the prespecified stratified
target bootstrap. It checks the locked chronology, original/bundled hashes,
sequence-only inference manifest, exclusion IDs/sequences and per-prediction
full-length/injection receipts. It does not rerun BLAST, regenerate structures
or independently score CIFs. The archived acceptance report separately records
the execution-side scoring and additional statistical audit.

Only pair-lDDT interaction Psi is the sole primary endpoint. The fixed
Train384/ESMC recipe was selected using prior observed results; the panel was
locked before new predictions. Other contrasts are secondary and unadjusted.
Intervals condition on the fitted models and rotations; neither complete
project exposure nor family/pretraining isolation is guaranteed.
