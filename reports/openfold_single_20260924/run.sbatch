#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=paper-of-repro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/shuang886/Folding/engramholder_repro_20260924/logs/%x_%j.log
set -euo pipefail
module load gcc/12.5
TASK_ROOT=/data/user/shuang886/Folding/engramholder_repro_20260924
export PYTHON_BIN=/data/user/shuang886/.conda/envs/fold/bin/python3.12
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export CUDA_HOME=/usr/local/cuda-12.8
unset PYTHONPATH
bash "$TASK_ROOT/package/reproducibility/openfold_single/setup_env.sh" "$TASK_ROOT/clean_env" "$TASK_ROOT/wheels" "$TASK_ROOT/assets/openfold.tar.gz"
export PYTHONPATH="$TASK_ROOT/clean_env/openfold"
PKG="$TASK_ROOT/package/reproducibility/openfold_single"
cd "$TASK_ROOT"
"$TASK_ROOT/clean_env/venv/bin/python" "$PKG/predict.py" --fasta "$PKG/example.fasta" --af2-weights "$TASK_ROOT/assets/params_model_3_ptm.npz" --esm-weights "$TASK_ROOT/assets/esm2_t12_35M_UR50D.pt" --out "$TASK_ROOT/prediction"
"$TASK_ROOT/clean_env/venv/bin/python" "$PKG/score.py" --prediction "$TASK_ROOT/prediction/prediction.cif" --reference "$PKG/example_reference.cif.gz" --mapping "$PKG/example_reference.json" --out "$TASK_ROOT/prediction/score.json"
