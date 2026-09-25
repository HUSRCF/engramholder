#!/bin/bash
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
set -euo pipefail
TASK_ROOT="$2"
PARENT_ROOT=/data/user/shuang886/Folding/engramfold/prospective_orientation_series_20260924
module load gcc/12.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$TASK_ROOT/staging:$PARENT_ROOT/vendor/openfold:$PARENT_ROOT/source/src:$PARENT_ROOT/source/scripts"
export CUDA_HOME=/usr/local/cuda-12.8
exec "$PARENT_ROOT/env_openfold/bin/python" -u "$TASK_ROOT/staging/$1.py" --root "$TASK_ROOT" --index "${SLURM_ARRAY_TASK_ID:-0}"
