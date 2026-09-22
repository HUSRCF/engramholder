# Bounded reproduction artifact

This directory contains actual adapter/operator source snapshots, a CPU-only
operator check, and fixed configuration records. The paper asset generator at
`../scripts/build_paper_assets.py` reconstructs numbers/tables/figures from fixed
score JSON and verifies 348 system-by-metric means against target records.

```sh
python -m pip install numpy matplotlib torch
python scripts/build_paper_assets.py
python reproducibility/operator_smoke.py
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

The compact archive deliberately does not contain model weights, all experimental
structures, ESM caches, all checkpoints, or a tested turnkey retraining launcher.
Those remain requirements for a complete training reproduction release. Do not
present this package as satisfying them. The original personal repository and
its Git history must not be included as an anonymous supplement.
