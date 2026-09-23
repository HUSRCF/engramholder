# Frozen-system independent validation v2

Relocked2026-09-18 with explicit user approval after v1 selection could not fill
its shortest stratum (14/24 from11423 records). No new target prediction or
score existed at amendment. Preserve v1 protocol, selection decisions and failure.
Only MSA-row coverage denominator changes; all other scientific rules remain.
Explicit user authorization:
new evaluation only, no training. This does not revise the failed UT development
primary endpoint, its untriggered bilinear follow-up, or either Engram gate.

## Fixed systems and endpoint

Exactly96 new chains, four length strata128–191/192–255/256–319/320–384,
24 chains each. Twelve existing task384 checkpoints (T/UT × factor/generic ×
20260917/18/19), one injected query baseline, official
protenix_mini_esm_v0.5.0 with native ESM2-3B. All14 instances use c4/s5,
one sample, seed101, FP32, no templates/MSA search/ligand input, no MC dropout,
no outcome-based sample selection. Official ESM follows its native no-MSA path;
our query/adapter systems preserve the existing injected OPM path and pair stack.
No historical best checkpoint selection and no new Engram/bilinear system.

Unique primary: T-factor minus T-generic at384, average three matched seed
differences within target, then equal-target mean.20000 percentile bootstrap
resamples of the96 targets, RNG seed20260918, two-sided95% interval. Continue
research gate requires mean>=0.02, lower bound>0, and all three per-seed panel
means>0. This tests the average of the frozen models, not the population of
retraining randomness. No gate fallback, extra targets or additional seeds.
UT, query, official ESM, length strata and auxiliary metrics are secondary.
No equivalence claim from a zero-crossing interval.

## Outcome-independent panel selection

Source: existing monomer_candidates.jsonl.gz metadata and raw experimental CIFs.
Only initial release<=2021-09-30, monomer_clean, one model, X-ray resolution<=2.5A,
standard20-AA sequence and lengths128–384. No parent temporal-test manifest or
coordinates are opened. Exclude old32 PDB IDs and sequences, and near homologs
to all32 old queries plus all non-query sequences in their actually used fixed
MSAs (conservative inclusion of development MSAs). No new homolog search.

Sort each length stratum by SHA256('engramfold-independent96-v1|'+target_id).
Take first24 eligible after mapping and sequence isolation, one PDB per panel.
Validate entity_poly_seq exactly matches catalog, model1 label-indexed CA
coordinates finite, >=90% sequence CA coverage, indices valid, and >=1 pair
within15A. Reference observed indices are then fixed for every system.

Near-homology exclusion: Biopython global affine alignment, match2/mismatch−1,
gap open−8/extend−1; first optimal alignment; >=30% identity among paired
columns, >=50 paired columns. For old32 queries and previously accepted panel
chains, retain >=70% coverage of the shorter sequence. For old MSA rows only,
require >=70% coverage of the CANDIDATE sequence. A sequence appearing in both
old queries and MSA rows retains the stricter query rule. Candidate is always
first argument; no symmetric swapping for this MSA-fragment test. Candidate
rank string remains engramfold-independent96-v1 (unchanged
from v1); no reseeding after the feasibility failure. This is
pairwise sequence isolation, NOT proof of remote-family separation or absence
of PLM/folding pretraining overlap. Chain bootstrap may understate unobserved
family dependence. Do not rebrand the panel as a family-held-out benchmark.
Save all selection/rejection decisions, exclusion sequence and source hashes,
reference hashes, checkpoint hashes, immutable protocol and manifest hashes
before computing any candidate prediction. If eligibility cannot yield96, stop
selection without relaxing rules or opening scores.

## Scoring, failures and cost

Primary is existing0–1 pair-averaged C-alpha lDDT, fixed reference CA mask,
15A reference-distance pairs, thresholds0.5/1/2/4A. This is not standard all-atom
or residue-averaged lDDT. Missing/nonfinite predicted CA contributes zero for
its reference pairs; no method-specific intersection mask. Record observed
coverage and prediction coverage separately. Catastrophic prediction failure
scores0 and stays in denominator96; failure rate reported per system. Identical
job may be resumed for an infrastructure interruption; preserve errors and do
not alter model/precision/inputs to rescue a target. Kabsch CA TM-like score is
secondary and explicitly not optimized TM-align. Score only after all prediction
attempts; no interim scores used to tune or extend panel.

Report model-load time separately; synchronized per-target PLM generation,
adapter/query-context, folding and feature-preparation times; GPU allocated
peak for each stage. Cold-feature cost includes PLM (weights already loaded);
process-cold startup separately includes model loading. Cached-PLM cost excludes
only representation generation. No cached embedding presented as free cold
inference. No aggregate speedup claim if measured stages are incomplete.

Engineering validation uses only old training chains before opening new panel
predictions; protocol/checkpoint lock precedes panel feature generation. A
neutral manuscript evidence outline may be written without predicted outcomes.
