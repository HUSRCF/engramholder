# OpenFold training-budget review — 2026-09-23

Read-only inspection of already completed historical OpenFold Train96/Train384
artifacts. No new training, prediction, fresh-panel access, or source changes.

## What the existing evidence establishes

- Both historical studies have 15 formal training logs; every run ends at step
  1536. The inspected directories contain checkpoints at 384, 768 and 1536,
  with no longer-step result found there.
- The formal runner evaluates structures after training, not after each saved
  checkpoint. No formal per-checkpoint Dev structure-quality curve was found.
  The Train96 study has six 384-step Dev8 learning-rate calibration evaluations;
  these are not convergence measurements of the final formal fits.
- The schedule concatenates independent random permutations of the complete
  training set. All 30 formal schedules were checked against logged target IDs:
  each Train96 target occurs exactly 16 times, and each Train384 target exactly
  4 times in 1536 updates. These are exact counts, not merely average passes.
- Some Train96 logs contain repeated step numbers from interrupted/resumed
  execution. For this inspection, the last record for each step was retained;
  the resulting steps were exactly 1 through 1536 and matched `schedule.json`.
  Raw log-line counts must not be interpreted as additional final-model updates.

## Observed training loss

Each entry is the arithmetic mean of the training loss over successive
nonoverlapping blocks of 384 steps: 1–384, 385–768, 769–1152, 1153–1536.
These are training losses on the scheduled targets, not held-out structural
scores or a fixed-target loss probe.

| Training set | Head | Seed | Four loss means |
|---|---|---:|---|
| Train96 | Native | 20260923 | 1.902450, 1.746653, 1.705566, 1.670370 |
| Train96 | Native | 20260924 | 1.905081, 1.748894, 1.687643, 1.655759 |
| Train96 | Native | 20260925 | 1.911045, 1.755077, 1.716503, 1.673935 |
| Train96 | G+ | 20260923 | 1.971792, 1.787765, 1.716809, 1.671297 |
| Train96 | G+ | 20260924 | 1.961571, 1.811734, 1.732189, 1.699978 |
| Train96 | G+ | 20260925 | 1.953542, 1.782870, 1.731932, 1.698798 |
| Train384 | Native | 20260923 | 1.890201, 1.769397, 1.745621, 1.723692 |
| Train384 | Native | 20260924 | 1.887360, 1.769794, 1.751271, 1.741501 |
| Train384 | Native | 20260925 | 1.906227, 1.774582, 1.740630, 1.722208 |
| Train384 | G+ | 20260923 | 1.941981, 1.782038, 1.742223, 1.726276 |
| Train384 | G+ | 20260924 | 1.936902, 1.801564, 1.765201, 1.726743 |
| Train384 | G+ | 20260925 | 1.946384, 1.790985, 1.744989, 1.719855 |

The final loss block remains lower than the preceding block in these runs.
There is insufficient evidence to declare convergence at 1536 updates.
This observation does not establish that longer training would improve test
quality, restore the head-by-rotation interaction, or isolate a data-size effect.
Training-set composition and repetition change together between these recipes.

## Exact sources inspected

Remote completed studies on `hpc3` (read through X570):

- `/data/user/shuang886/Folding/engramfold/cross_backbone_20260921/openfold/formal/*/training.jsonl`
- `/data/user/shuang886/Folding/engramfold/cross_backbone_20260921/openfold/formal/*/schedule.json`
- `/data/user/shuang886/Folding/engramfold/cross_backbone_20260921/openfold/calibration/*/evaluation.json`
- `/data/user/shuang886/Folding/engramfold/openfold_train384_20260921/openfold/formal/*/training.jsonl`
- `/data/user/shuang886/Folding/engramfold/openfold_train384_20260921/openfold/formal/*/schedule.json`

Checkpoint and evaluation filename inventories were read in the same formal
directories. Numerical loss rows above use `native_s20260923/24/25` and
`gplus_s20260923/24/25`; exposure checks included all 15 systems per study.

Local source snapshots:

- Permutation schedule（实验仓库 `/home/husrcf/Code/onestepfold/engramfold/src/engramfold/experiments/cross_backbone_protocol.py`）, lines 22–24.
- Formal training, checkpoints and final evaluation（实验仓库 `/home/husrcf/Code/onestepfold/engramfold/scripts/run_openfold_study.py`）, lines 61–96.
- [Train96 rotation execution lock](x570-engramholder-52dde3b/reproducibility/openfold_rotation_execution_lock.json).
- [Train384 Stage A execution lock](x570-engramholder-52dde3b/evidence/openfold_esmc_A_execution_lock.json), which identifies its historical Train384 source directory.

This is a review note recording the completed inspection, not a new execution
lock, a convergence test, or a manuscript evidence migration.
