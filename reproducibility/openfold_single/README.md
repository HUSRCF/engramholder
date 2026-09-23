# One-target OpenFold prediction and scoring

This entry reconstructs a frozen OpenFold Train96/ESM2 Native adapter prediction
from a FASTA, without reading reference coordinates or reusing PLM/AF2 features.
It exports all predicted heavy atoms to mmCIF; a separate process scores the
paper's primary fixed-reference Cα pair-lDDT. It does not retrain an adapter or
reproduce the full experiment matrix. The example is the first previously
observed engineering target, 3gxb_A, with the first native training seed.

## Assets and sources

- `adapter.pt` contains only the released writer tensors and compact metadata,
  exported from the fixed 1536-update checkpoint. `asset_manifest.json` records
  both hashes. No trained weight is changed during export.
- OpenFold source is fixed at
  [be2ec1841f16c966c65ae0e7599ebbadc725757d](https://github.com/aqlaboratory/openfold/tree/be2ec1841f16c966c65ae0e7599ebbadc725757d).
  `setup_env.sh` checks the source archive hash before building its CUDA extension.
- Obtain the official AF2 `params_model_3_ptm.npz` using the pinned OpenFold
  [parameter download script](https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/scripts/download_alphafold_params.sh).
  The official archive is `https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar`.
  Extract `params_model_3_ptm.npz`; the predictor rejects any different hash.
  OpenFold code and AF2 weights retain their upstream licenses and attribution;
  see the [official OpenFold notice](https://github.com/aqlaboratory/openfold#copyright-notice).
- Obtain [ESM2-35M](https://github.com/facebookresearch/esm) weights from
  `https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt`.
  The exact hash is checked. No contact-prediction regression weights are required.
- `example.fasta` and the separate reference derive from public PDB 3GXB,
  [RCSB entry](https://www.rcsb.org/structure/3GXB). The mapping preserves full
  entity-sequence indices and the reference hash used in the original scoring.
  The reference file and mapping are never arguments to the prediction process.

The heavyweight AF2 and ESM2 weights are not bundled. They may be staged from an
existing authorized download if their hashes match; feature caches may not be
substituted for the FASTA-to-feature computation in this reproduction.

## Fresh environment and prediction

Run from the artifact/repository root. Requirements: Linux x86-64, Python 3.12,
CUDA toolkit 12.8, a compatible C++ compiler (tested GCC 12.5), and an H100 for
the tested configuration. Other hardware is not validated by this example.
A fresh environment uses several GB of disk. The commands below do not need
MSA search, template databases, JAX, historical experiment roots or cluster paths.

```bash
bash reproducibility/openfold_single/setup_env.sh "$PWD/openfold-clean"
export PYTHONPATH="$PWD/openfold-clean/openfold"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
openfold-clean/venv/bin/python reproducibility/openfold_single/predict.py \
  --fasta reproducibility/openfold_single/example.fasta \
  --af2-weights /path/to/params_model_3_ptm.npz \
  --esm-weights /path/to/esm2_t12_35M_UR50D.pt \
  --out "$PWD/openfold-example-output"
openfold-clean/venv/bin/python reproducibility/openfold_single/score.py \
  --prediction openfold-example-output/prediction.cif \
  --reference reproducibility/openfold_single/example_reference.cif.gz \
  --mapping reproducibility/openfold_single/example_reference.json \
  --out openfold-example-output/score.json
```

Use a new output directory for each attempt. The example expects one standard
amino-acid sequence and generates a full-length prediction. `run.json` records
four trunk/OPM calls, frozen-backbone identity, actual module paths, freshly
computed feature identity and the output hash. `score.json` starts from the CIF,
not from a saved scalar or the prediction NPZ. The NPZ remains available for
coordinate replay checks. TM-score is outside this minimal scoring example.

## Offline installations

Where compute nodes lack internet, download the pinned packages on a connected
machine into a wheelhouse (including dependencies and the `ihm` source archive),
and download the pinned OpenFold source tarball. Then run:

```bash
bash reproducibility/openfold_single/setup_env.sh \
  "$PWD/openfold-clean" /path/to/wheelhouse /path/to/openfold-source.tar.gz
```

The same clean venv is used; `include-system-site-packages` is false. The tested
installation freeze is `environment.freeze.txt`; `validation.json` records the
passed replay. The reproduced pair-lDDT is 0.568206229860 versus historical
0.568259935553 (absolute difference 0.000053705693, preset tolerance 0.0001).
`expected_prediction.cif` and `expected_score.json` are output examples, never
inputs to the predictor. Initial missing-dependency attempts are retained in the engineering
report; they must not be represented as successful predictions. A successful
one-target replay supports this entry only, not full retraining reproducibility
or identical results across all hardware and environments.
