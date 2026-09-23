# Native directions and first-order adaptation: bounded study v1

Authorized 2026-09-19. This is a NEW study after the closed interface/access/budget
studies. Their endpoints, checkpoints and conclusions are unchanged. Mini only;
no Tiny, Train384, new loss, memory, learning-rate search or projection solver.

## Hypotheses and operators

At the existing query anchor, use Wtilde = D/(D+eps) W, D=508.5:
T = Wtilde[da outer bq + aq outer db], Q = Wtilde[da outer db].
Four constructions inject Uq+T+Q, Uq+T, Uq+R(T+Q), Uq+RT. Only the residual
is rotated. The query depth=1 affine remains unchanged. Both orders have the
same parameter count and per-channel rank bound min(L,64); deleting Q does not
reduce this bound or prove a regularization mechanism.

Three fixed orthogonal matrices are generated in CPU float64 from Gaussian
matrices by QR with positive R diagonal, then stored as float32. Dedicated
rotation seeds 20261001/02/03 do not consume training RNG. Matrix bytes/hashes
are locked before any training scores. Rotation preserves decoder singular
values and residual/Jacobian Gram geometry ONLY at matched parameter states,
not along independently learned trajectories.

## Fixed training matrix

Training seeds 20260923/24/25; same initial writer per paired seed. All task-only,
lr5e-5, AdamW(weight_decay=1e-4,betas=.9/.999,eps=1e-8), clip1, FP32,
one target/noise draw per update. Freeze ESM2-35M and Protenix Mini default0.5.0.
Reuse the existing native denoising/distogram loss and last-recycle gradients.
Keep configured bond loss and report actual bond masks (historically empty).

Train24: native full/tangent x3 seeds =6 runs; rotated full/tangent x3 rotations
x3 seeds =18 runs, all to384. Native six additionally continue with the same
optimizer/RNG contract to1536. Train96: native full/tangent x3 seeds =6 fresh
runs to1536. Total30 independent fits plus6 continuations,25344 updates.
No score-based early stopping, checkpoint/LR selection, replacement seeds or
extra rotations. Save/evaluate native384/1536 and rotated384. Development
uses Train24 or Train96 and the original Dev8. Report both fixed1536-update
and fixed16-pass (Train24@384 vs Train96@1536) comparisons. Equal updates do
not establish equal measured compute. Timing includes separate data transfer,
training, checkpoint, feature export and inference costs.

## Joint outcome-blind data lock

Existing Train24/Dev8 membership is taken exactly from interface cache_v1.
Use only the existing catalog records released by2021-09-30, monomer_clean,
single model, X-ray<=2.5A, standard20AA, length128..384; exact entity mapping,
finite CA coverage>=90%. No parent reserved test manifest/post-cutoff reads.
Exclude all old32+observed96 PDBs and query near-homologs. Use the frozen old
MSA exclusion sequence corpus. Global Biopython alignment: match2,mismatch-1,
open-8,extend-1, first optimal alignment; >=50paired, identity>=.3,
coverage>=.7 of shorter sequence for queries/new-chain comparisons; oldMSA
coverage>=.7 of candidate length (v2). Query duplicates retain stricter rule.

Select confirmation FIRST,24 per128-191/192-255/256-319/320-384 stratum,
ordered by SHA256('native-direction96-v1|confirmation|'+target_id). Then select
18 additions per stratum by SHA256('native-direction96-v1|train_addition|'+id),
excluding confirmation and all already selected additions by the same pairwise
rule. Retain old24 unchanged; obtain nested Train96. One new chain per PDB.
Save all decisions, raw-coordinate/catalog/reference/protocol/source hashes.
If any stratum cannot fill, stop selection WITHOUT relaxation; do not substitute
a smaller confirmatory panel or train set. This isolates adaptation/development
under sequence rules, not remote families or foundation pretraining.

New task-only cache contains sequence features, query factors/update and
supervised structure labels, no teacher update. All new runs use CPU-backed,
per-target GPU loading, including Train24. This is a versioned new cache contract.
Confirmation targets are never training/selection inputs. New feature caches
may differ in index hash but must match PLM model/checkpoint/layer/sequence.

## Independent confirmation and analysis

Freeze all declared checkpoints before any confirmation predictions. Evaluate
24 Train24@384 variants,6 nativeTrain24@1536,12 nativeTrain96@384/1536,
query and officialMiniESM =44 instances x96 =4224 predictions. c4/s5,1sample,
inference seed101,FP32. Fixed reference mapping/masks; missing predictions zero,
no best-of samples. Wait for ALL prediction attempts before aggregated scores.
Training failure invalidates a complete primary comparison; never score a
failed training run as a zero-quality rotated control to claim superiority.

Unique primary: native tangent Train24@384 minus the mean of the THREE rotated
tangent Train24@384 models, paired within each of THREE training seeds, then
averaged within target. Report mean,20000 target-bootstrap95% interval
(seed20260926), all training-seed means and all rotation means. Support gate:
mean>=.02, lowerCI>0, all3seed and all3rotation aggregate differences>0.
Intervals condition on fitted models; repetitions are not additional proteins.
Full-vs-rotated-full, tangent-vs-full and data-size/budget effects are secondary
descriptive comparisons, cannot replace the primary. No equivalence claim from
an interval crossing zero; any noninferiority description must use the fixed
.01 margin and identify it as secondary. Reuse audited CA pair-lDDT; supplementary
residue-average CA-lDDT and fixed-correspondence full-length TM-score retain
their supplementary status. Sequence-cluster sensitivity does not confer family
labels; a one-cluster interval is not meaningful.

## Existing-checkpoint diagnostics

Record T/Q energies,rQ=||Q||^2/(||T||^2+||Q||^2+eps), cosine and inner product
for the three historical selected Factor seeds at96/192/384/768/1536 on32old
targets. Undefined zero-vector cosine is null. This is descriptive; no mechanism
claim or model selection. No post-hoc Q-removal folding or ridge projection in
the18h scope: prioritize complete trained controls and confirmation.

## Deadline and reporting

Abstract deadline2026-09-19T11:59:00Z (19:59HKT); result inclusion cutoff
2026-09-19T07:59:00Z (15:59HKT). Draft truthful abstract from existing evidence
immediately. Add new claims only after their complete runs/audits. Incomplete
work continues for the full paper; no partial-panel/seed positive conclusion.
The authorized compressed study supersedes the prior closure only for this
specific bounded scope. No scientific outcome triggers extra experiments.
