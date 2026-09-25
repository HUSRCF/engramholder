#!/bin/bash
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --nice=0
set -euo pipefail
TASK_ROOT="$2"
module load gcc/12.5
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$TASK_ROOT/repetition:$TASK_ROOT/staging:$TASK_ROOT/vendor/openfold:$TASK_ROOT/source/src:$TASK_ROOT/source/scripts"
export CUDA_HOME=/usr/local/cuda-12.8
case "$1" in
 engineering)
  for phase in continuous prefix resume; do
   "$TASK_ROOT/env_openfold/bin/python" -u "$TASK_ROOT/repetition/engineering.py" --root "$TASK_ROOT" --index "$SLURM_ARRAY_TASK_ID" --phase "$phase"
  done ;;
 formal) exec "$TASK_ROOT/env_openfold/bin/python" -u "$TASK_ROOT/repetition/run.py" --root "$TASK_ROOT" --index "$SLURM_ARRAY_TASK_ID" ;;
 *) exit 2 ;;
esac
