#!/bin/bash
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
TASK_ROOT="$2"
PARENT_ROOT=/data/user/shuang886/Folding/engramfold/prospective_orientation_series_20260924
module load gcc/12.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$TASK_ROOT/staging:$PARENT_ROOT/vendor/openfold:$PARENT_ROOT/source/src:$PARENT_ROOT/source/scripts"
exec "$PARENT_ROOT/env_openfold/bin/python" -u "$TASK_ROOT/staging/analyze.py" --root "$TASK_ROOT" --mode "$1"
