# Completed budget and injection-schedule follow-ups

These are two distinct studies on already observed panels. Run
`python scripts/verify_budget_schedule.py` from the artifact root to reconstruct
the saved-score comparisons. It checks all 3,552 OpenFold and 9,984 Protenix
rows, paired target arrays, original bootstrap streams, seed/rotation marginals,
Dev curves and receipt/source bindings. It additionally compares all 4,800
Protenix parent/Query replays with the original Fresh192 score grid.
This is score-level verification, not new training or CIF rescoring.

## Protenix: one fixed budget extension

Twenty-four Train384/ESMC-600M Full models continued from update1536 to3072,
restoring weights and AdamW state at constant learning rate5e-5. Fresh192 was
already observed. The sole primary endpoint is final Psi; its paired change
from1536 is secondary. Neither a null change interval nor positive final Psi
establishes convergence or equivalence. The historical checkpoint did not
contain all RNG states, so this is not claimed to reproduce an uninterrupted
historical trajectory bit for bit.

The `C_` prefix in archived model names denotes the ESMC feature configuration,
not the orthogonal compensator used in another study. No compensator is added.
The source snapshot records continuation/optimizer handling; it still relies
on the original experiment runtime, features and backbone weights.

`evidence/pt_budget_records.json` is the unchanged pre-recovery score table.
Its hash equals the final completion receipt's score-table hash. The recovery
filtered reference metadata to the sealed eight Dev IDs (the original metadata
also contained the 384 training chains). No predictions or score definitions
changed. Original receipts, the isolated recovery wrapper and execution-side
review are retained here; the original execution lock is not rewritten.

## OpenFold: closed single-rotation scope

The original protocol planned72 fits. The user closed execution at36 completed
fits: First/All/Last × four cells × three seeds, with only R1=20261001.
R2/R3 are not pending and are not launched. The archived amendment/summary keep
their original `SINGLE_ROTATION_INTERIM_NOT_FULL72` machine label; it denotes
the restricted scope, not an unfinished36-fit study. No claim is made to have
completed the originally planned three-rotation primary analysis.

All schedules were retrained with gradients through four passes and one common
Train/Dev-selected learning rate5e-5. Historical final-pass-only training and
fixed-model deployment ablations are different studies. These results change
the training/inference schedule together and do not isolate a feedback mechanism.
The table's All condition has positive Psi despite an unestablished Factor
Native−R1 effect, because rotated G+ outperforms native G+. First has larger
Psi and lower absolute quality. Both facts are retained in the paper.

## Included records and boundaries

`import_manifest.json` binds every imported snapshot. `evidence/pt_budget_*`
and `evidence/of_schedule_*` contain score grids, results and locks.
The study subdirectories preserve protocols/reports, implementation snapshots,
calibration and recovery evidence. The paper's active versioned source lock
also hashes this folder. Anonymous packaging records original and redacted
hashes separately, and the verifier respects that mapping.

We do not bundle the full experiment environment, all parent/endpoint weights,
raw CIFs or a complete retraining launcher. Execution-side coordinate, frozen-
parameter and restart checks are reported as such, not re-executed by this verifier.
