# Bounded reproduction artifact

This directory contains actual adapter/operator source snapshots, a CPU-only
operator check, and fixed configuration records. The paper asset generator at
`../scripts/build_paper_assets.py` reconstructs numbers/tables/figures from fixed
score JSON and verifies 348 system-by-metric means against target records,
144 target-level OpenFold interactions, six Protenix G+ contrasts, and 36
full-inference source/norm cells and contrasts. The active v12 input lock contains
266 source files and 1,851 generated numerical fields. All 1,830 previously
integrated values, display formats and scientific source mappings are unchanged;
21 fields add the bounded Atlas propagation table and Native-only calibration.
The v11 additions cover Protenix Fresh192 and the same-seed compensation repeat. The earlier additions
cover compensation (including explicitly post-hoc target-distribution summaries),
signed permutations and developmental checkpoint studies.
Absolute structure scores are displayed to four decimals (168 formatting-only
changes); effects, intervals, p-values and diagnostic errors retain their original
precision. The interaction violins use paired per-target effects, not bootstrap
replicates; their mean markers and intervals retain the original estimands.
The reconstruction prediction study remains a negative predictive test. Its verifier reconstructs 2,688 score records,
51 three-metric contrasts and the two exact 40,320-permutation tests (primary X,
exploratory D). Eight rotations are the prediction units; neither association
was established. Target bootstrap is conditional sensitivity, not more units.
The compensation verifier reconstructs 21 original contrasts from 3,552 unique
E1/E2 records and checks that both rotation averages were specified in the
sealed written design and analysis code. These are observed-panel mechanism
contrasts, not a new-target primary confirmation. The signed-permutation and
checkpoint verifier reconstructs 18 contrasts from 3,600 records and 72 contrasts
from 1,160 records, respectively, preserving each original bootstrap stream.
Ideal signed-coordinate AdamW compatibility is not a demonstrated full-model
FP32 trajectory equivalence; the Dev8 curves do not establish convergence.
The [anchor-intervention verifier and source bundle](anchor_intervention/README.md)
add all 1,824 model–target score records and reconstruct 42 paired contrasts.
The expected mean benefit of live residual anchors was not established. This
independently retrained intervention preserves the live baseline and recycling,
uses four round-matched query-only reference anchors, and contains no C.
It additionally reconstructs 216 A66 and 18 Protenix Train96 contrasts from
14,112 and 2,400 score records, including target bootstrap and model marginals.
A66 prespecifies ESMC Factor-minus-Rotated on Confirm96-B per backbone; its
interactions and other follow-ups have unadjusted intervals. These old-panel
results do not replace Fresh96. It also
verifies all 2,400 Fresh96 score records and three-metric contrasts, plus the
post-hoc Train384-minus-Train96 change of the full interaction on both panels.
Fresh96 does not establish its sole primary interaction; its positive Factor
contrast is not substituted for that endpoint.
The [Protenix Fresh192 bundle](protenix_fresh192/README.md) adds 4,800 fixed-model
predictions on 192 new targets. The verifier reconstructs 24 contrasts and the
prespecified stratified bootstrap; the sole primary pair-lDDT interaction is
positive. It does not replace the OpenFold result or guarantee family/pretraining
isolation. The [compensation repeat](e2_retraining/README.md) adds nine completed
same-seed training executions, 864 predictions and 21 reconstructed contrasts.
The rotated-arm benefit remains positive, but the additional benefit over Native
is not re-established. Original no-C baselines are fixed; executions are not
pooled as extra seeds. The execution reused the environment and PLM caches, so
it does not complete the separate fresh-feature full-prediction replay.
Literal numerical references in the manuscript and generated tables are checked
against the generated keys; an unknown key also raises a LaTeX PackageError.
The [Atlas boundary bundle](atlas_followup/README.md) retains all 20 two-chain
propagation instances, unrounded coordinates and the full propagation figure.
Its verifier rebuilds recorded state-summary aggregations, prediction-distance
changes and 18 calibration contrasts from 960 scores. Neither hidden-state
differences nor Native-only calibration establish a rotation interaction or an
explanation for absent adaptation gains.

```sh
python -m pip install numpy matplotlib torch gemmi
python scripts/build_paper_assets.py
python tests/test_numeric_keys.py
python tests/test_openfold_followups.py
python tests/test_single_prediction_score.py
python tests/test_e1_prediction.py
python tests/test_e2_intervention.py
python tests/test_signed_and_curves.py
python scripts/verify_anchor_intervention.py
python tests/test_anchor_intervention.py
python scripts/verify_protenix_fresh192.py
python tests/test_protenix_fresh192.py
python scripts/verify_e2_retraining.py
python scripts/verify_atlas_followup.py
python reproducibility/operator_smoke.py
python reproducibility/compensation/smoke.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build iclr2027_conference.tex
```

Run from the archive root. A LaTeX installation with the packages used by the
official ICLR style is required for the PDF. No folding GPU is needed for these
commands. The original checks used NumPy 1.26.4, Matplotlib 3.10.8 and Torch 2.9.1;
no universal dependency compatibility has been established.

The operator test covers zero initialization for Factor/G+, native Full/Tangent
rotation paths, and the actual OpenFold G+ output-layer reparameterization. It
is not an end-to-end folding or training reproduction. OpenFold/Atlas runtime
source is included to expose the integration boundaries, but depends on the
corresponding upstream repositories, licensed weights and prepared data.

`fixed_configs.json` is a checked recipe summary, not a complete data-preparation
or cluster-launch configuration. Historical execution locks identify the sources
used in the runs; `source_manifest.json` identifies the accompanying working
source snapshot. These are different provenance claims.

The compact archive includes one Native adapter and a tested complete-prediction
entry in [`openfold_single/`](openfold_single/README.md). On one already observed
target, a fresh isolated Python environment recomputed PLM features from FASTA,
generated a full-atom CIF, and scored it in a separate process. The preset replay
tolerances passed; the validation record and complete engineering history state
its limits. The example does not rely on an author-specific feature cache.

Pretrained AF2/ESM2 weights, all experimental structures, the remaining adapter
checkpoints and a full retraining environment are not bundled. Obtain the
pretrained weights from the pinned upstream sources under their terms. This
one-target entry does not reproduce the full matrix or establish portability to
all devices. `fixed_configs.json` remains a historical recipe summary, while the
single-target directory contains its own executable setup/configuration.
The personal repository and its Git history must not be included in an anonymous
supplement. Binary/metadata anonymization still requires author review.

The [shared-channel compensation example](compensation/README.md) executes the
sealed E2 writer and optimizer on synthetic CPU OPM inputs, verifies initialization,
parameter groups, mask/rotation placement and joint gradients, and explains the
full-model integration boundary. This is a runnable operator-level intervention,
not an additional complete-protein training reproduction.
