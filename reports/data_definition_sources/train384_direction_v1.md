# Train384 direction expansion — locked scope v1

Authorized2026-09-20: user selects Train96→Train384, native versus rotated full
Factor on DiamondHill. Separate from active hpc3v4; no changes to its endpoints.

Primary comparison: native minus mean of three original rotations, Mini full
Factor, nested Train384,1536updates,three original training seeds20260923–25.
Rotations20261001–03;12new fits. Same zero initialization, frozenESM2-35M,
frozenMini,queryanchor,decoder depth508.5,taskloss,learningrate5e-5 and sequence
schedule implementation. No teacher pretraining,no LR search,no6144step extension.

Train384 contains all existingTrain96 plus72new chains per length stratum
128–191,192–255,256–319,320–384. Deterministic hash order
train384-direction-v1|target_id. Pre2021-09-30 existing catalog; original
experimental quality,monomer,reference mapping and90%CAcoverage rules unchanged.
Exclude all296known observed queryPDBs for new additions; sequence exclusions
include currentTrain96/Dev8,Confirm96-A/B and documented oldMSArows. Query
homology uses30%identity,50aligned positions,70%shorter coverage; oldMSArows
use70%candidate coverage. New additions mutually screened with same query rule.
No unknown family annotation becomes evidence of novelty. Parent temporal test
untouched. Screen at most2048candidates/stratum within2hours; no relaxation.
If quotas cannot be filled, report blocked before training.

Score all12frozen1536checkpoints on existing observedConfirm96-B using sealed
c4/s5,reference masks,sample selection and failure rules. New scores are an
observed-panel data-scale follow-up, not blind confirmation. Report allseeds,
rotations,absolute scores andchain-bootstrapCIs. PrimarymetricCApair-lDDT;
TM-score/residueCA-lDDT fromsameCIFs are supplementary.

Secondary data contrast: newTrain384 versus existing matchedTrain96/1536models,
with strict initialization/loss/decoder/backbone provenance checks. This changes
data composition and mean exposure perchain(16to4passes), not pure samplecount
causality. Do not compare to a different checkpoint's bestselectedstep.
If historical pairing cannot be verified, report withinTrain384comparison only;
no silentbaseline rerun or historicalscore substitution.

CPU-backed/per-target native training cache; no confirmation labels. Exporter
must validate disjointness,hashes andnestedness, and retain existingTrain96
cache tensors bitwise. New training cache separated from diagnostic caches.

Resources: DiamondHill atmost8logicalMI250devices,CPU selection64workers (up to
128only if measured beneficial),4Torchthreads/GPU. NoPrecision. Data preparation
and12fits precede1152CIF evaluations. Estimates~15–20GPUhtraining plus preprocessing
andscoring, before measuredpilot; no promised walltime until selection/cache ready.
No newjobs beyond9hours frombranchstart; unfinishedwork explicitly retained.
Formaldata/model/configlocks precede model results. No stopping by score sign.
