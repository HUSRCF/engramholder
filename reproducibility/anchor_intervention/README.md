# Residual-anchor intervention

This directory documents the completed OpenFold anchor intervention, internally
E3 in the prospective series. It is distinct from the older local-loss diagnostic
also called E3. No channel compensator C is used here.

The only intervention replaces the residual's live factors by the corresponding
query-only factors for each of four recycle passes. The live OPM baseline,
48 Evoformer blocks, structure module and complete recycling remain active.
Nine reference-anchor models were trained from the original zero-output
initialization; the live-anchor models are the accepted E1 references.

## Run the bounded checks

From the artifact/repository root, with NumPy and PyTorch installed:

```sh
python scripts/verify_anchor_intervention.py
python tests/test_anchor_intervention.py
python scripts/build_paper_assets.py
```

The verifier reconstructs all 42 three-metric contrasts from 1,824 unique
model–target records: 864 new reference-anchor predictions and 960 exact reused
E1 predictions. It checks the 96 paired targets, three seeds, two selected
rotations, 20,000 synchronous target draws, absolute means, seed/target arrays
and the identity `T_A = B_R_live_minus_Q + (N_Q - N_L)`. It verifies source and
receipt bindings without reading the neighboring execution repository.

The five CPU tests check effect signs and common-control pairing, reject invalid
arrays, and exercise the actual archived hook: the baseline stays live, forward
calls use references 0/1/2/3, backward recomputation uses reference 3, and an
incompatible mask is rejected. This fixture is not a full OpenFold prediction.

## Readable evidence and source identities

| File | Role |
|---|---|
| `protocol.md` | Original pre-execution intervention, endpoints and budget |
| `archived/query_anchor.py` | Exact live-baseline/reference-residual hook and reference capture |
| `archived/references.py`, `e3_common.py` | Reference certification, identities and model setup |
| `archived/run.py`, `parent_core.py` | Formal runner and inherited model/optimizer integration |
| `archived/analyze.py`, `scoring.py` | Original aggregation and scoring definitions |
| `archived/job.sh`, `cpu.sh`, `smoke.py`, `test_query_anchor.py` | Original launch and engineering checks |
| `acceptance.md`, `runtime_audit.py`, `statistical_readback.py` | Execution-side audit and its implementation |
| `engineering_amendment.md` | Preserved failed exact-trajectory check and bounded replacement checks |
| `per_target.csv` | Complete target effects exported by the accepted audit |

The `evidence/` directory at the artifact root contains:

| JSON snapshot | Contents |
|---|---|
| `e3_anchor_records.json`, `e3_anchor_new_records.json` | All 1,824 records and the 864 new-record subset |
| `e3_anchor_summary.json`, `e3_anchor_statistical_readback.json` | Original analysis and independent execution-side readback |
| `e3_anchor_execution_lock.json`, `e3_anchor_complete.json` | Locked recipe/source hashes and formal completion receipt |
| `e3_anchor_runtime_audit.json`, `e3_anchor_acceptance.json` | Raw-coordinate audit and acceptance receipt |
| `e3_anchor_references.json` | 192 cache certifications, covering 384 reference forwards |
| `e3_anchor_engineering_amendment.json`, `e3_anchor_resume_diagnosis.json` | Exact-resume failure, diagnosis and amendment |

All 11 JSON snapshots and 17 archived protocol/source/audit files are inputs to
`paper_sources.v10.lock.json`. Earlier locks are retained. The anonymous artifact
redacts personal host/path strings and binds both original and bundled hashes in
`bundled_source_provenance.json`; numerical records and algorithms are preserved.
Archived scripts and Markdown retain their original relative paths and historical
status statements. Use the mapping above for their bundled locations. They are
readable execution snapshots, not portable launch commands for a new environment.

## Result identity and limits

Both predeclared average quantities have negative point estimates and intervals
including zero. The expected mean benefit of live anchors is not established.
The reference-anchor gap is smaller in point estimate while Native decreases;
this is not the useful joint improvement observed in the separate C intervention.
The two rotations have opposing responses. Their secondary intervals are
unadjusted, and there is no declared between-rotation heterogeneity or equivalence
test. Confirm96-B was already observed. Target intervals condition on the fixed
fitted models and selected rotations.

The execution-side acceptance audit rescored all 1,824 primary scores from raw
coordinates and parsed stored TM-score outputs. The commands above independently
reconstruct statistics from bundled scores and check the archived audit's
consistency; they do not rerun that coordinate audit or a TM-score binary. The
raw coordinate/reference caches, nine trained checkpoints, upstream weights and
complete training environment are not included. No full-training or full-quality
replay is claimed. This study neither retests the E1 predictor nor tests whether
C's benefit depends on the anchor source.
