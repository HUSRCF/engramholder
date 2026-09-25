#!/bin/bash
#SBATCH -p debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
TASK_ROOT="$2"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$TASK_ROOT/repetition:$TASK_ROOT/staging:$TASK_ROOT/vendor/openfold:$TASK_ROOT/source/src:$TASK_ROOT/source/scripts"
exec "$TASK_ROOT/env_openfold/bin/python" -u "$TASK_ROOT/repetition/$1.py" --root "$TASK_ROOT"
