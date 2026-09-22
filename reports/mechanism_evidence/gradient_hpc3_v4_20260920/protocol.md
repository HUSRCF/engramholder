# H100 v4 proposal: reachable directions versus learned updates

Status: AUTHORIZED AND ACTIVE, 2026-09-20, user “可以，开始”.
Execution start03:26:37UTC, reporting deadline12:26:37UTC. The v3
results, failed historical gates and sealed structure endpoints remain unchanged.

## Question

Does the trained adapter actually use the useful directions available in its
native update space? v3 established an observed-panel local oracle advantage,
but its correlations with trained structural gains were unsupported. Increasing
only the number of projections would not answer this gap.

Use hpc3 H100, at most eight simultaneous single-GPU workers. No Precision,
Tiny, new adapter fitting, new rotations or shallow-MSA experiments. A's twelve
annotation gaps remain unresolved; do not label this study family-held-out.

## Scope and hierarchy

### E1: noise robustness and loss components, full observed96

Reuse the fixed Confirm96-B manifest and all three existing ordinary rotations.
Use nine native-sampler noise conditions per target: preserve v3's three seeds,
add six prospectively locked seeds. Store actual sigma, augmented coordinates,
noise and RNG, rather than relying on reseeding across branches. Recompute all
nine on H100 with the same code version; historical three-condition results
remain a separately reported reference.

At the query-only anchor, freeze/detach the first three recycles and compute one
common gradient per target/noise. Save total, weighted denoising and weighted
distogram gradients, verify additivity. Project the total into four spaces using
v3's common Gram scale, ridge1e-4 and FP64 solver/tolerance. For the two loss
components, project separately as secondary analyses; do not change the training
loss or select the more favorable component as the main endpoint.

Use only the predeclared primary amplitude1e-3 for bulk +/- interventions.
Retain v3's three-amplitude grid for the initial two-chain smoke and eight-chain
numerical audit. All bulk FD exceptions remain listed; report fixed-branch and
native rematching objectives separately. Do not silently exclude failed pairs.

Counts: 96×9=864 common gradient conditions; 3,456 primary total-gradient
solves and 6,912 signed final-cycle forwards. Three component projections total
10,368 solves, including the total. Native sampler draws are paired, not chosen
by sigma or score; noise-bin summaries are descriptive only.

### E2: learned direction use — primary new comparison

Use all twelve archived Mini/Train24/tangent/384 models: three native seeds and
three training seeds crossed with the three ordinary rotations. This inventory
is supported by existing metric records; checkpoint paths, hashes and matching
configurations must be verified before execution. No unavailable checkpoint is
replaced with a newly trained or selected model.

For each E1 query-only condition obtain each frozen student's actual final-cycle
residual delta under the same features and anchor. Include its rotation exactly
as deployed, and verify zero-residual replay. At this stage preceding recycles
remain query-only: this is a controlled common-state intervention, not the
student's complete inference trajectory.

For nonzero residuals set d=delta/||delta|| (do NOT flip the sign using labels).
Record:

- norm ratio ||delta||/||Uq||;
- signed local utility q_learned=−gᵀd/||g||;
- projection-oracle utility q_oracle=gᵀv/(||g||||v||), v=P_lambda g;
- q_learned/q_oracle only when the denominator is numerically resolved;
  this is a relative diagnostic, not an exact bounded efficiency ratio,
  because ridge is a shrinkage map rather than an orthogonal projector;
- actual native loss decrease at equal amplitude eta=1e-3||Uq||, and at
  the model's actual delta. The latter is nonlinear and does not receive a
  local derivative interpretation. Record increases as well as decreases.

Primary new endpoint: the native-minus-mean-rotations difference in actual,
equal-norm normalized native loss decrease, averaged over nine noise conditions
and paired training seeds within each chain. One primary endpoint; oracle
alignment, actual-scale effects and components are secondary.

Counts: 96×12×9=10,368 student-direction cases, two forward evaluations each
(equal norm and actual scale):20,736 cached last-cycle forwards. Baselines and
gradients are shared with E1; do not repeat backbone backward per student.
Student passes depend on x, not the noise draw where the architecture permits
safe caching; establish equivalence before caching.

### E3: full trained-state check — fixed first 24 targets, optional budget tier

Select six targets per existing length stratum by stable target-ID order,
without looking at outcomes. Use the twelve archived Mini/Train96/full/1536
models: three native, nine rotated. These models already have structure scores.

Run each model's actual four-recycle route, caching its own first three states.
At its learned fourth-cycle state, compare retaining its residual versus
removing only that final residual. Freeze preceding states and noise within
this pair. Use three native noise conditions. No claim that this comparison
isolates geometry: preceding learned states differ across models.

Counts:24×12×3=864 model-state/noise conditions, each with a learned-state
baseline and one residual-removal replay. Report task loss effect, absolute
loss and residual scale, alongside existing matched structure scores.
This asks whether the common-state conclusion survives a more realistic
learned context. It does not generate new CIFs or require another full sampling
benchmark. There is no assumption that a trained full model's Jacobian equals
the query tangent map; E3 does not reuse the latter as its local derivative.

E3 starts only if E1/E2 coverage and a time-based pilot forecast leave two hours
for analysis before the hard end. Trigger depends on resources/time, not effect
sign. If skipped, report NOT RUN rather than extending the deadline.

## Controls, statistics and stopping

- Same frozen Mini, ESM features, masks, loss weights, c4 contract and pair stack
  as the sealed studies. No factor loss, confidence optimization or optimizer.
- Preserve diagnostic-only label cache; never feed confirmation labels into
  student inference or training caches. Labels enter only offline loss analysis.
- Chain is the statistical unit; aggregate noises, seeds and rotations first.
  Report20,000 chain-bootstrap intervals and seed/rotation marginals.
- Existing96 has already been observed. No independent-confirmation or
  family-generalization claim. Correlations with old structure scores are
  secondary observed associations, including negative and null values.
- No quality-threshold search, per-target sign flipping, favorable rotation
  selection, post-result change of ridge or loss weights, or replacement targets.
- Numerical failures remain counted, with full-cohort coverage and partial-set
  summaries explicitly distinguished. Re-run deterministic infrastructure
  failures with identical inputs; do not retune based on scientific outcomes.
- Stop dependent E2/E3 if learned-update replay is not validated by T+2h.
  Report the precise blocker; do not replace it with new training.

## H100 execution and budget

Nine hours from actual execution start is a proposed cap, not a guaranteed ETA.
Slurm availability is not assumed. Reuse the known fold environment and
LAYERNORM_TYPE=torch; record Torch/CUDA/GPU/config/source hashes. Environment
repairs are routine implementation work, separate from scientific changes.

T+0–0.5h: snapshot protocol, checkpoint/config hashes, target order, added seeds;
check queues and stage only missing files. Do not overwrite v3 outputs.
T+0.5–1.5h: two-chain E1/E2 replay, FD and learned-residual tests; pilot two
length extremes for time and peak-memory prediction.
T+1.5–6.5h: E1/E2 at up to8 H100 workers. Assign complete target bundles,
length-squared balanced; do not assign each rotation to a separate worker.
E1 and E2 for a target can share cache/noise/gradient state; no cross-target
batching required. Use one process/GPU, FP32 backbone and FP64 solves, no
unvalidated mixed precision. Request8 CPUs and64GB host RAM per worker;
measure first, increase only if needed. Avoid global128-thread BLAS oversubscription.
T+6.5–7h: optional E3 only if prospectively forecast to finish byT+7h.
T+7h: stop dispatching new target bundles; drain within the bounded allocation.
T+8–9h: synchronize outputs, coverage/frozen/hash audit, statistics and report.
Incomplete work atT+9h stays incomplete; no outcome-dependent extension.

Core forward budget:27,648 cached last-cycle forwards, plus864 baseline/gradient
conditions and component backward work;10,368 projection solves. This is larger
than v3, but not27,648 full c4/s5 folds. Pilot measurements determine actual
GPU-hours including model loading and I/O. Maximum eight-GPU nine-hour envelope
is72allocated GPU-hours, not a predicted consumption. Reduce concurrency if
I/O or memory contention negates throughput; never shrink target coverage based
on measured scores. Optional E3 is the first scope removed for time.

## Interpretation locked in advance

1. Oracle and learned equal-norm effects both favor native: supports use of
   locally useful directions, still not a complete explanation of lDDT gains.
2. Oracle favors native but learned directions do not: reachable-space advantage
   is insufficient; shared predictor learning/dynamics remain central.
3. Only actual-scale effects differ: investigate magnitude as a candidate factor,
   without claiming direction alone explains the difference.
4. Common-state and trained-state effects disagree: mechanism is context-dependent.
5. No supported correlation with structural gains: keep that negative result;
   do not search further metrics until a favorable association appears.

Deliver execution_lock_v4.json, checkpoint_inventory.json, runtime_smoke.json,
per-target/noise/model records, coverage/FD/freeze audit, analysis.json, two
scientific figures and results.md. Manuscript changes follow completed evidence,
not this proposal. Do not open a new confirmation panel during this run.
