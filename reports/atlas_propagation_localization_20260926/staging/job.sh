#!/usr/bin/env bash
set -euo pipefail
ATLAS_LOCALIZE_ROOT=/media/PM982/engramfold/runs/atlas_propagation_localization_20260926
ATLAS_DEPS_ROOT=/media/PM982/engramfold/folding_e2e_20260921/deps
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$ATLAS_LOCALIZE_ROOT/source/src:$ATLAS_DEPS_ROOT/atlas_training_compat:$ATLAS_DEPS_ROOT/atlas_d3"
export HIP_VISIBLE_DEVICES="$2"
cd "$ATLAS_LOCALIZE_ROOT"
exec timeout 3500 /media/PM982/engramfold/folding_e2e_20260921/venvs/atlasfold-rocm/bin/python -u -m engramfold.experiments.atlas_propagation_localization --root "$ATLAS_LOCALIZE_ROOT" --index "$1"
