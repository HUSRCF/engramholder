# AtlasFold Native-only posttraining calibration — 2026-09-25

## Question and scope

Does a learned, shared orthogonal output-channel transformation improve an already
trained Native factor adapter more than spending the same 512 additional updates
on its original head? Does it also improve on unadapted AtlasFold, rather than
merely repair a harmful adapter? This is an observed-panel followup, not fresh
blind confirmation. There are no rotated arms, so no rotation interaction or
direction-cost mechanism claim is identified by this experiment.

## Fixed experiment

Use the three archived ESM2 Train96 Native checkpoints at step1536, seeds
20260923/24/25. ESM2 is fixed before any new outcomes; these parents all used
the same MI250 backend. Keep AtlasLM, folding weights, insertion point, features,
loss, train set and inference protocol unchanged.

| Branch | Trainable state | Starting state | Updates per seed |
|---|---|---|---:|
| C_only | 8001 skew coordinates of mean-preserving C in SO(127) | Parent head frozen; C=I; new AdamW moments | 512 |
| head_only | Original 423168-parameter Native head | Parent parameters and complete AdamW moments, step1536 | 512 |

The C map acts on the complete 128-channel adapter residual before adding the
frozen baseline. It does not rotate the whole pair state. Its matrix exponential
and multiplication use FP32 outside BF16 autocast, followed by the original
residual dtype; identity must preserve the actual residual exactly. The head and
live-factor computation remain differentiable where required.

Continue the original 96-target permutation stream through global step2048;
verify its first1536 target IDs against each parent schedule. Each paired branch
uses the same next512 IDs and the historical per-forward Torch seed rule. Parent
checkpoints lack saved RNG states: this is controlled continuation from recorded
weights/optimizer, not a claim of bitwise continuation of unrecorded RNG.

Use the original AdamW settings, clipping norm1. Initial head LR=1e-4*q, C LR=1e-3*q;
head decay=.01, C decay=0. Choose one shared q from {.3,1} using only Dev8:
first parent seed, both branches, each q,64 updates; select the largest pooled
absolute pair-lDDT across the two branches, choosing .3 for ties within1e-6.
Formal fits restart from their parent with this common q and use the final512
checkpoint without early stopping. Do not pick separate branch-specific rates.

## Gates, inference and scoring

Before calibration, each branch must pass C=I/original-head forward equivalence
on the first and longest training targets,8 updates, and a4+4 checkpoint resume.
Frozen model/head hashes, finite connected gradients, optimizer/RNG restoration,
and C geometry are checked. Whole-trajectory equality after resume is diagnostic;
zero gradients are retained and recorded rather than treated as proof of failure.
Losses, separate C/theta norms and zero-gradient parameter names are logged.

Only Confirm96-B (the existing evaluation144 manifest's `confirm96` subset) is
used for formal evaluation. The old Train/Dev selections retain their own
metadata; no new date-cutoff rule is imposed on inherited chains. No Length48
extension is included. All96 targets and failures stay in the denominator.

Predict ten systems afresh with the same runtime: unadapted Query, three parent
Native models, and six new models. Sequence-only inference manifests, Python
file-open auditing and a guard on the used Gemmi parser block reference paths
in prediction. This is a checked code path, not an OS sandbox. Scoring starts
only after all ten systems are terminal and all identities/hashes pass. Training
labels are of course available only during training; Dev labels only select q.

Prediction preserves original settings: seed1, one sample, four requested
recycles (five actual trunk calls), MLM probability.15,20 diffusion steps for
these short chains. Main metric is fixed-mask pair-lDDT; residue-averaged CA-lDDT
and TM-score normalized by full sequence length are descriptive robustness checks.

Primary contrast: C_only minus head_only. Also report C_only minus parent,
head_only minus parent, both new branches minus Query, and parent minus Query.
Average three paired fitted seeds before bootstrapping targets20,000 times with
seed20260925. Show per-seed means and95% intervals conditional on the fitted
models. Secondary intervals are descriptive and unadjusted.

Interpretation: superiority to head_only alone establishes a relative benefit of
this allocation of512 updates. If C_only still falls below Query, call it repair
or mitigation, not improved adaptation. Strong practical support requires positive
C_only−head_only and C_only−Query with compatible per-seed and auxiliary results.
Null results bound this protocol, feature recipe and budget; they do not prove
that AtlasFold cannot benefit from channel calibration.

## Resource and stopping policy

Use DiamondHill HIP6/7 (PCI8e/93; monitor GPU4/5), disjoint from the current
Protenix controller's HIP0–5. Isolated source and execution lock; no edits to
existing runs/checkpoints or manuscript. Stop automatically on engineering gate
failure and preserve evidence. Do not silently change activation, seeds, loss,
targets, budget or clipping to obtain successful outcomes.

Budget:6×512=3072 formal updates,256 Dev-calibration updates,32 smoke updates,
960 formal and32 Dev predictions. Historical runtime suggests about5–7 hours
on two MI250 devices, including initialization and scoring; this is an estimate.
Implementation and results belong in this experiment's files, never system memory.
