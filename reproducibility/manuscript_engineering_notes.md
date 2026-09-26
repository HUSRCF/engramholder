# Engineering provenance moved out of the manuscript

This note preserves operational material removed or condensed during the
2026-09-26 editorial scope revision. These are historical records from the
manuscript snapshot, not a new execution, new acceptance decision, or a claim
that all experiments were independently reproduced.

The paper retains the scientific methods, selection and exclusion rules,
predeclared endpoints, results, missingness handling, mixed-backend limits,
compensation-repeat numerical sensitivity, and the single-R1 reporting
amendment. Routine acceptance checks, recovery histories, source/receipt
inventories, and delivery status belong in artifact documentation.

The excerpts below are exact pre-edit text from
`appendices/implementation_and_evidence.tex` (starting snapshot `05c2110`).
Line numbers refer to that snapshot. A replaced passage can contain scientific
context that remains in condensed form in the paper; the excerpt is retained
in full so the editorial move does not erase its engineering history. LaTeX
macros refer to the manuscript's generated values, whose definitions and
scientific source locks were not changed by this revision.

The removed portable example is documented in
[`openfold_single/README.md`](openfold_single/README.md). Compensation execution
and numerical diagnosis are documented in
[`e2_retraining/README.md`](e2_retraining/README.md) and
[`compensation_diagnosis/README.md`](compensation_diagnosis/README.md).
Anchor execution is documented in
[`anchor_intervention/README.md`](anchor_intervention/README.md), Fresh192 in
[`protenix_fresh192/README.md`](protenix_fresh192/README.md), and the budget and
schedule follow-ups in
[`budget_schedule_followups/README.md`](budget_schedule_followups/README.md).

## Routine recipe verification

Original lines 97–99.

```tex
backbones use native losses and inference budgets: equal update counts do not
mean equal computation. Frozen-parameter hashes, zero-initialization replay,
output-head gradients and subsequent encoder gradients are checked.
```

## Membership-audit scope

Original lines 151–152.

```tex
are operational sequence exclusions, not family labels. This audit verifies
stored membership, lengths and metadata; it does not rerun sequence searches.
```

## Writing chronology and unmatched historical comparison

Original lines 158–161.

```tex
distinctions remain unchanged when the paper presents the later OpenFold
interaction first. The separate Train24 G+ study used its own seeds and
head-specific development-selected rates; its results cannot fill matched
Train384 cells.
```

## Public writing history

Original lines 197–198.

```tex
The public writing history is separate from these scientific locks; it does
not retrospectively change endpoint status.
```

## Paper-generation checks and historical gates

Original lines 220–223.

```tex
20,000 draws on each panel. The paper generation script reads locked intervals
and independently reconstructs the six new G+ contrasts with the same rule;
it does not select a new resampling procedure. Original primary gate definitions
remain in the historical records; later comparisons do not replace them.
```

## Generated-value source mapping and rendering checks

Original lines 241–247.

```tex
Table~\ref{tab:scope_full} provides the exact direction contrasts underlying
Figure~\ref{fig:scope}. The accompanying score artifact retains every target
and model instance; \texttt{generated/cell\_sources.json} maps each generated
number to its evidence field and records input hashes. A single script produces
both LaTeX values and figures and independently checks completed raw system
means before rendering. No structure prediction or scoring is performed by
that script.
```

## Differential-study acceptance status

Original lines 320–326.

```tex
sequence predictor's learning problem. A later parameter-space differential
study passed independent writer and gradient-chain checks but encountered
responses near downstream numerical resolution; its original S0 remained
unpassed. A separately declared finite-step Train8-to-Dev8 study produced
measurable responses but did not establish mean positive transfer or a native
advantage. These distinct outcomes are retained rather than merged into one
successful mechanism claim.
```

## Local-diagnostic completion and source audit

Original lines 361–371.

```tex
All 864 core conditions and 864 retention conditions were completed. At the primary
oracle step, 3398/3456 dynamic finite differences meet the 5\% relative criterion;
58 exceptions remain included. These checks concern oracle directions, not
every learned direction. Independent scalar reconstruction reproduces the
archived target means and intervals. Five of six current v4 source files match
the historical hashes; the main analysis script differs, so the reconstruction
uses a separate verifier. Conflicting historical start-time fields are preserved,
not treated as independently certified chronology. For provenance, the v4
records call the oracle, learned-update and retention diagnostics E1, E2 and
E3, respectively; those internal labels are separate from the subsequent
prediction, compensation and anchor-intervention studies.
```

## Full-inference prediction completion

Original lines 381–384.

```tex
reference-structure mask. All 720 pure and mixed predictions are rerun on the
same backend; none fails. The mixed residuals are inference-time interventions
and need not correspond to outputs of the trained factor head or states
encountered during training.
```

## Residual preaudit acceptance

Original lines 435–436.

```tex
All 288 nonzero, finite residuals pass the preaudit. The model is fixed
Protenix-Mini Train384/Full/1536; no target supervision enters residual formation,
```

## Full-inference hashes, preflight repairs, and reconstruction checks

Original lines 465–473.

```tex
The completion audit records all CIF hashes and frozen parameters, paired query
hashes and RNG states, and 432 mixed-norm/direction checks. Actual random-tensor
hash pairing is checked on the two engineering targets only. There are no formal
failures or HIP retries. Initial preflight failures, their repaired fields and
CPU/device comparison issue, and their superseded locks remain separately
recorded. The formal protocol is frozen after these repairs and before scoring.

All 720 score records reconstruct the 36 metric-by-cell/contrast summaries,
target arrays, seed and rotation marginals to within $10^{-12}$. The primary
```

## Paper verifier execution limits

Original lines 478–479.

```tex
proteins. Full-predictor computations and CIF scoring are not rerun by the
paper's numerical reconstruction script.
```

## Engineering audits and recovery records

Original lines 481–503.

```tex
\section{Engineering audits and recovery records}
\label{app:engineering}
Zero-residual replay, input/parameter gradients, frozen hashes, pairing, and
complete-sequence outputs are audited independently of quality-based model
selection. Backend reruns are numerical checks, not new biological samples.
In the AtlasFold final evaluation, broken cross-machine absolute symlinks caused
720 recorded engineering failures across the first five runs. Repairs retained
the same final checkpoints, reused the available short-chain CIFs, and generated
missing outputs under independent recovery reports. The final 2160 predictions
were complete; the original attempts were not erased or selected by quality.
Post-start supplementary checks in the original length study retain their
actual chronology rather than being relabeled as prelaunch checks.

For matched Protenix G+, three two-step prefixes passed before formal
continuation; those steps count within each completed 1536-update run.
All 432 formal predictions succeeded without HIP retries. Audits checked the
complete training order, paired noise levels, frozen-parameter and artifact
hashes; score contrasts were independently reconstructed. Confirm96-B uses
MI250 for both sides. Length48 reuses historical H100 Native/Rotated predictions
and new MI250 G+ predictions. Short-chain replay was exact, while a 750-residue
replay had relative coordinate L2 error 0.00032018, below the predeclared
$10^{-3}$ engineering tolerance. This limited replay does not establish a
numerical error bound for the entire long-chain panel.
```

## Feature-substitution prediction completion totals

Original lines 520–524.

```tex
The same observed Confirm96-B and Length48 targets supply 144 scores per new
fit: 4,752 new predictions with no failures. The combined score archive contains
7,200 records, including historical systems and baselines. This is distinct
from the original Train96 four-cell study; its primary endpoint is ESMC
Native--Rotated on Confirm96-B, not a newly selected interaction endpoint.
```

## Four-cell follow-up completion heading

Original lines 603–603.

```tex
\section{Completed Protenix and AtlasFold four-cell follow-ups}
```

## Train96 membership hashes and completion totals

Original lines 609–613.

```tex
Train96 and Dev8 manifest hashes match the OpenFold experiment. This matches
training membership and update count, not backbone loss or computation cost.
The 1,152 new Confirm96-B predictions complete the 2,400-record factorial plus
query baseline; no Train96 Length48 factorial is added. Its primary is the
direct pair-lDDT interaction on the already observed Confirm96-B panel.
```

## Four-cell aggregate execution totals

Original lines 619–622.

```tex
Protenix retains $5\times10^{-5}$ and $10^{-4}$. All 9,504 new predictions
complete, combined with 4,608 historical score records. No new quality-based
hyperparameter search is introduced. Atlas's native AtlasLM-3B is retained;
ESMC replaces only the added adapter features. As in OpenFold, this changes
```

## Backend provenance heading

Original lines 651–651.

```tex
\paragraph{Backend and engineering provenance.}
```

## Four-cell engineering failures and verifier coverage

Original lines 657–668.

```tex
replication. The execution archive retains Atlas gradient-assertion failures,
checkpoint recovery, hidden-layer saturation events, the amended diagnostic
policy and H100 Torch API compatibility edits. Final matrix completion is not
claimed to mean an absence of engineering failures. No targets, model arms,
final checkpoints, activations or learning rates were selected by these scores.

The paper verifier reconstructs all 216 A66 and 18 Train96 metric/contrast
combinations from 14,112 and 2,400 score records respectively, including
pairing, bootstrap intervals and seed/rotation marginals. It neither reruns
folding nor rescores CIFs. Source hashes, original locks and the scoring
amendment are retained; this verification does not certify an end-to-end
retraining reproduction.
```

## Fresh96 locks, replay, and acceptance inventory

Original lines 751–760.

```tex
This study reused the 24 OpenFold Train96/ESM2-35M adapters and query-only
baseline without further training. The model lock preceded target selection;
execution and scoring locks preceded the first new prediction. All 2,400
model--target predictions completed before scoring, with zero failures. Fifty
engineering predictions on two previously observed targets were separate from
the formal panel. Their coordinates and three scores replayed the historical
outputs exactly. The collection audit checked all 24 checkpoint hashes, 2,400
prediction hashes and 328 source/lock bindings; score-array reconstruction
independently recovered the reported contrasts and marginals. These checks do
not reproduce the CIF scoring or certify every historical access event.
```

## Fresh96 exposure-inventory bug and failed attempts

Original lines 769–772.

```tex
An initial global-alignment rule failed to fill the first stratum and rejected
all eight fixed shuffled negative controls. A separate exposure-inventory bug
had first mislabeled 256 synthetic controls as real queries; correcting that
bug did not resolve the alignment failure. Both failed attempts are archived.
```

## Fresh96 search-audit counts

Original lines 782–786.

```tex
candidate length for historical teacher rows; HSPs are not merged. The archived
selection audit checked 474 searches and reported no known ID/PDB/full-sequence
overlap. This is sequence-screened new-target validation relative to auditable
project exposure, not the identical Confirm96-B screening implementation,
strict family isolation or exclusion from foundation-model pretraining.
```

## Post-hoc aggregation validation

Original lines 831–835.

```tex
After the ESM2 Train384 four-cell control was complete, a subsequent analysis
computed $q_i=\Psi_{384,i}-\Psi_{96,i}$ for every shared target. Model identities,
three seeds, three rotations and complete target coverage were checked before
aggregation. The analysis uses 20,000 paired whole-target bootstrap draws
(seed 20260924), retaining all model cells for each sampled target. These are
```

## Portable one-target reproduction entry

Original lines 855–885.

```tex
\section{A portable one-target complete-prediction entry}
\label{app:single_repro}
The accompanying \texttt{reproducibility/openfold\_single} entry includes a
public FASTA, the fixed Native adapter exported without changing writer tensors,
a pinned environment/source setup script, a label-free predictor and a separate
CIF scorer. It was executed on an H100 using a newly created Python 3.12 venv
with system site-packages disabled, Torch 2.7.1+cu128 and a newly compiled,
revision-pinned OpenFold extension. Host Python, CUDA 12.8 and GCC 12.5 remain
explicit prerequisites; this is not an operating-system/container reproduction.

The first previously observed engineering target, 3gxb\_A, and the first Native
training seed were fixed for this check. ESM2 layer-12 features were recomputed
from its 184-residue FASTA, without historical feature caches. All 184 residues
were predicted through four trunk and selected OPM calls, with backbone and
adapter frozen. All predicted heavy atoms were exported to mmCIF; only the
separate scorer read experimental coordinates. The published output was also
rescored from CIF on the local machine with the same result. The verification
record reports the preset feature/coordinate tolerances, the score difference,
file hashes and actual import paths. This validates one model, target and
primary metric, not the complete experimental matrix, all supplementary metrics
or end-to-end adapter retraining.

The archived engineering history includes two missing-dependency starts before
prediction, followed by a successful run. A copied reference symlink was replaced
by a hash-identical ordinary file and independently rescored. The current writer
source matches the verified Fresh96 inference snapshot; its later rotated-G+
branch differs from the initial Train96 training snapshot, without changing this
Native Factor computation. These provenance distinctions are retained rather
than describing all historical sources as byte-identical. Pretrained AF2/ESM2
weights are obtained separately from upstream under their original terms and
verified by hash; they are not bundled in the compact artifact.
```

## Internal prediction-study identifiers

Original lines 890–894.

```tex
This subsequent study (internally E1) tests a candidate predictor of the
post-training Native--Rotated score difference, distinct from the earlier v4
oracle diagnostic also called E1. It fixes
OpenFold/Train96/ESM2/Full, 1536 task updates and Confirm96-B, an already
observed panel. Eight new rotations were locked before diagnostic measurements;
```

## Prediction-test completion and score-archive totals

Original lines 922–926.

```tex
All 24 trainings and 2,304 new predictions completed with no prediction failures.
The outcome $Y_R$ is the mean Native-minus-retrained-Rotated pair-lDDT over the
96 targets and three seeds. With 384 audited historical Native/query outputs,
the complete score array contains 2,688 records. None is an additional
independent rotation for the prediction test.
```

## Reconstruction diagnostic engineering-resolution gate

Original lines 943–947.

```tex
the primary predictor. The $X$ span is \result{e1_x_span}, exceeding the
predeclared $10^{-5}$ engineering resolution gate. Its large absolute error
reflects partial, finite-budget reconstruction, not a lower bound on
representational error. The prediction hypothesis was not supported; this was
not a failure to resolve variation in the diagnostic.
```

## Auxiliary scorer precision-check failure and repair

Original lines 977–983.

```tex
The scoring audit retained an auxiliary-check precision failure: casting
prediction coordinates from FP32 to FP64 changed a thresholded pair credit in
four records, without changing the fixed reference pair set. The repaired
auxiliary check uses the main scorer's original input precision and separately
records the FP64 comparison. Main scores, their inputs and the statistical
rules were unchanged. The bundled verifier reconstructs outcomes and exact
tests from score arrays; it does not rerun CIF scoring or training.
```

## Signed-control completion totals

Original lines 998–1002.

```tex
This observed-panel OpenFold/Train96/ESM2/Full/1536 follow-up trains three
signed transforms across three seeds in Factor and G+: 18 fits and 2,592
successful predictions. Reusing six audited Native/G+ fits and query outputs
gives 3,600 records. Confirm96-B pair-lDDT $\Psi$ is primary; Length48 and other
metrics are secondary and unadjusted, not new-target confirmation.
```

## Small-head validation status

Original lines 1018–1021.

```tex
Small-head VJP and optimizer checks passed, but a strict correspondence of
complete FP32 training trajectories was not established. Non-significant G+
differences are not equivalence tests, nor do they rule out optimization or
numerical contributions to Factor's behavior.
```

## Saved-node prediction completion count

Original lines 1050–1052.

```tex
The 1,160 successful predictions comprise $48\times3\times8$ adapted predictions
and eight shared query outputs. These are observed-development diagnostics,
not a new confirmation panel or a basis for selecting the final checkpoint.
```

## Compensation internal identifier and completion status

Original lines 1086–1093.

```tex
The separately locked compensation study (internally E2) uses the same
OpenFold/Train96/ESM2/Full/1536 recipe as the prediction study in
Appendix~\ref{app:prediction_e1}. It adds a shared channel map to Native and to
r20270107/r20270104, selected as low/high reconstruction error before the new
rotations' task training. Selection was not based on their later structure
scores. There are three paired seeds per condition: nine fits and 864 new
Confirm96-B predictions, all complete without prediction failures.
The original Native and rotated models provide the paired no-map controls.
```

## Compensation sealed-design and code assurances

Original lines 1146–1150.

```tex
scores of the compensated models. The original sealed design explicitly defines equal-weight averages of
$B_R^{C}$ and $T_C$ and the prediction that both are positive, with both rotations
and Native's own change also reported. The sealed analysis code implements
these averages before outcomes. These are prespecified mechanism contrasts,
not a new-target confirmation or a newly declared sole-primary endpoint.
```

## Compensation diagnostic repair, receipts, and operator-demo checks

Original lines 1196–1212.

```tex
The engineering record preserves the earlier diagnostic dtype failure at the
96th update and the authorized restart from checkpoint zero. All nine final
runs have complete 1--1536 logs and 16 periodic checks below the original
$10^{-3}$ tolerance; the maximum orthogonality and mean-direction errors are
approximately $1.67\times10^{-5}$ and $2.18\times10^{-6}$.
These are constraint diagnostics, not full-model error bounds. Final checkpoint,
receipt and all 864 prediction hashes were audited. No structure score was
used to change the chosen rotations, training budget or retry rule.

The artifact includes the sealed channel module, original live OPM writer,
optimizer construction and exact rotation arrays. A CPU example executes their
zero initialization, shared-map injection and joint gradients on a synthetic
frozen OPM fixture; it checks masks, the unchanged baseline and parameter groups.
This makes the intervention executable at the operator level, without claiming
a replay of full OpenFold structure training or its quality gains. A later
complete training execution is reported separately below; it does not turn
the CPU fixture into a full-prediction reproduction entry.
```

## Repeat feature-cache and environment verification wording

Original lines 1219–1221.

```tex
four-pass live-anchor configuration were retained. The repeat reused the
verified H100 software environment and hashed ESM2 caches; it was not a new
installation or a fresh FASTA-to-feature replay. The no-$C$ models are the
```

## Compensation repeat execution-agreement table and acceptance runs

Original lines 1226–1267.

```tex
\begin{table}[!htbp]\centering\footnotesize
\caption{Compensation repeat: recorded common conditions and limits of the
comparison. The parent execution lock, repetition lock, archived wrapper,
run receipts and subsequent starting-state audit distinguish recipe agreement
from checked tensor identity and engineering replay. These checks do not
causally explain the full-training score difference.}
\label{tab:e2_execution_conditions}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{.15\linewidth}p{.43\linewidth}p{.36\linewidth}@{}}\toprule
Aspect & Recorded agreement & Remaining limit \\\midrule
Data / order & Original Train96, hashed input files and seed-derived task schedule;
1536 updates. & No claim of identical numerical states along the formal trajectories. \\
Initialization & Same three seeds; zero head output, $C=I$; a later audit finds
identical saved writer tensors, optimizer states and Torch/CUDA RNG states.
& Starting-state equality does not ensure equal numerical trajectories. \\
Source & Same parent lock and accepted diagnostic-repair runner/amendment;
the repeat wrapper is separately hashed. & Early formal logs differ; the original
intermediate tensors were not retained to identify their first divergent operation. \\
Weights / features & Parent hashes bind frozen AF2 weights, backbone state,
rotation arrays and cached ESM2 inputs. & Features are reused, not freshly extracted. \\
Software / device & Parent contract checks Torch 2.7.1+cu128, H100, TF32 off
and eight CPU threads. & A full comparison of driver and kernel runtime states
is not supplied. \\
Optimizer & Original AdamW groups: head learning rate $10^{-4}$, decay 0.01;
$C$ rate $10^{-3}$, no decay; global clipping at 1. & Matching hyperparameters
does not establish matching trajectories or convergence. \\
Checkpoint replay & Six engineering conditions restore parameters, optimizer,
RNG, learning rates and data position exactly; saved-state forward checks pass.
& Independent 32-step trajectories differ; this is not a full-trajectory
equivalence test. \\
\bottomrule\end{tabular}\end{table}

Six engineering conditions (I/two rotations, with/without $C$) first completed
384 updates across continuous and interrupted paths. Parameters, optimizer,
random states and data position were restored exactly; independent FP32
training trajectories were not required or found to be bitwise identical.
The nine formal fits then completed all 13,824 updates and 864 predictions,
without failed predictions. After all attempts ended, coordinates were exported
to full-atom CIF, parsed independently, and rescored. The execution audit records
exact masked FP32 coordinate recovery and identical values for all three
scores. This is a completed training execution and coordinate-scoring check;
public fresh-feature replay remains a distinct deliverable.
```

## Numerical-diagnosis audit heading

Original lines 1269–1269.

```tex
\paragraph{Post-hoc starting-state and numerical audit.}
```

## Ambient worker random-state record

Original lines 1274–1277.

```tex
Python/NumPy global RNG states were not recorded in these historical
checkpoints. Matching early log scalars did not establish matching hidden
tensors. The new workers also had different ambient Python/NumPy states;
their recorded Torch/CUDA states and actual input hashes matched.
```

## Numerical-diagnosis source hashes and artifact directory

Original lines 1285–1287.

```tex
gradients, parameters and optimizer tensors. Diagnostic summaries, source
hashes and capture boundaries accompany the artifact in
\texttt{reproducibility/compensation\_diagnosis/}.
```

## Repeat evidence-package and verifier inventory

Original lines 1321–1326.

```tex
The evidence bundle preserves both score arrays, the repetition protocol,
code hashes, original-baseline bindings, per-fit receipts and the CIF audit.
The numerical verifier reconstructs all 21 contrasts and their seed/target
marginals from saved scores, separately from the archived CIF audit. The
repeat neither validates the failed reconstruction predictor nor tests
compensation at reference anchors.
```

## Anchor-study internal E3 naming

Original lines 1333–1335.

```tex
reduces the Native--Rotated score difference. It is internally called E3 in the prospective
series, distinct from the earlier diagnostic E3 in
Appendix~\ref{app:mechanisms}. It contains no channel compensator $C$.
```

## Accepted anchor baselines and execution totals

Original lines 1363–1365.

```tex
The accepted $L$ models are reused from the prediction study. The 864 new predictions and
960 original query/Native/Rotated predictions yield 1,824 score records on
the already observed Confirm96-B panel; no target or rotation is replaced.
```

## Sealed field-name bookkeeping

Original lines 1376–1376.

```tex
$B_R^{C}=S(R+C)-S(R)$; only display notation changes, not sealed field names.
```

## Anchor-cache checks, engineering amendment, and reconstruction scope

Original lines 1454–1473.

```tex
\paragraph{Execution record and reproduction scope.}
All nine training runs, 13,824 updates and 864 new predictions completed with
no formal prediction failures. The 192 training/evaluation reference caches
were certified by 384 query-only forwards at two seeds, with equal factors
and coordinates and preserved RNG state. Training and evaluation caches
remain separate. The final reference is pinned during backward checkpoint
recomputation; 27 saved-node reference replays passed. The engineering
amendment retains an earlier failed bitwise next-update check and separately
checks exact saved-state/forward replay and a common-gradient AdamW update;
it does not claim strict full-model trajectory equivalence. Formal scientific
settings and source files were unchanged.

The archived execution audit independently rescored 1,824 raw-coordinate
primary scores and parsed the saved TM-score outputs. The accompanying
artifact bundles all 1,824 score records, the protocol, source snapshots,
audit receipts and analysis definitions. Its verifier reconstructs all 42
paired contrasts, target arrays, seed marginals and intervals, and checks
the identity above. CPU tests exercise the actual anchor hook's preserved
baseline, round selection and backward replay. These local checks do not
regenerate coordinates, rerun the TM-score binary or reproduce full training.
```

## Fresh192 execution-host inventory and selection repair

Original lines 1494–1499.

```tex
The exposure inventory was refreshed from four execution hosts, explicitly
including Fresh96 and later prediction, compensation and anchor-intervention
targets. Historical conservative exclusions were retained. Newly collected
search-only candidates and directory pseudonyms were corrected before new
target prediction; this correction did not change search thresholds or the
ranking salt. The final selection applies Fresh96's accepted BLAST v2 rules:
```

## Fresh192 search-audit counts and lock bindings

Original lines 1509–1514.

```tex
read model scores. The archived audit reports no ID, PDB or exact-sequence
overlap with the recorded exclusions and independently rechecks 1,195
searches. These claims concern the auditable inventory, not certified
complete project history, strict family separation or foundation-model
pretraining exclusion. The 192 sequences, label correspondence, model hashes,
execution recipe and analysis were locked before new folding predictions.
```

## Fresh192 execution heading

Original lines 1516–1516.

```tex
\paragraph{Execution and endpoint.}
```

## Fresh192 engineering replay criterion and formal completion

Original lines 1520–1527.

```tex
path and FP32 loading; folding uses the fixed MI250 path. Fifty engineering
predictions on two already observed targets were separate from the panel.
Their maximum relative coordinate discrepancy was below the prior $10^{-3}$
criterion. No target coordinates or observed-residue mask enter inference.
All $25\times192=4,800$ formal predictions completed successfully with full
length and four injections verified; uniform scoring followed completion.
The predefined failure rule would retain failed predictions at zero credit,
with no target replacement, crop or altered denominator.
```

## Fresh192 verifier and completion-receipt inventory

Original lines 1580–1587.

```tex
The accompanying evidence includes the score grid, sequence and reference
manifests, exposure exclusions, protocol, selection/execution/model locks,
source snapshots and completion receipts. The verifier independently
reassembles all 24 metric--contrast combinations, target arrays, seed/rotation
marginals and the stratified bootstrap. It also checks identity bindings,
prediction coverage and the recorded lock/completion chronology. This
score-level reproduction is distinct from rerunning BLAST searches, generating
new coordinates or rescoring CIF files.
```

## Atlas engineering-check artifact pointer

Original lines 1599–1600.

```tex
adaptation gains nor a rotation interaction. Full per-pass records, coordinates,
propagation plots and engineering checks are in the supplementary artifact.
```

## Atlas completion adjective

Original lines 1614–1614.

```tex
A separate completed Native-only calibration continued each parent for 512
```

## Atlas prediction completion count

Original lines 1622–1625.

```tex
Neither establishes a benefit. All 960 predictions completed; intervals use
20,000 target bootstrap draws after averaging three fitted seeds, with the
Query comparison descriptive and unadjusted. These results do not establish
information redundancy or motivate a specific alternative interface.
```

## Atlas replay-record pointer

Original lines 1644–1645.

```tex
The supplement provides all fixed-node scores, secondary contrasts and replay
records.
```

## Budget extension aggregate update count

Original lines 1653–1655.

```tex
and train to step3072: 36,864 additional updates. The original checkpoints
lack complete RNG states, so this is not a claim of bitwise equivalence to
uninterrupted historical training. No compensator is added, and the final
```

## Budget-extension engineering checks and Dev aggregation repair

Original lines 1703–1712.

```tex
The study completed 9,408 Fresh192 predictions (both model budgets plus Query)
and 576 Dev8 predictions at 1536/2304/3072, with no failures. Another 100
predictions and 288 updates belonged to isolated engineering checks. All 4,800
parent/Query replay score records match the original three metrics exactly.
The first aggregation failed because Dev reference metadata also contained
the training targets; the isolated repair selected only the sealed eight Dev
IDs, sequence identities and order. All 9,984 saved score records retained
their hashes. The supplement preserves the original execution lock, repair,
completion receipts and independent score reconstruction. Increasing scores
and three Dev nodes do not establish convergence.
```

## Schedule scope-amendment execution language

Original lines 1730–1737.

```tex
The original lock planned three rotations and 72 fits. After the first wave
completed, computational-budget constraints limited the final scope to its
36 fits: three schedules, four cells, three seeds and the first prespecified
rotation R1=20261001. The reporting amendment was recorded before unified
scoring of these predictions; the remaining two rotations were not launched.
We report the single-R1 analogue of the planned All--First comparison,
not completion of the original three-rotation primary matrix. Original locks
and the amendment remain in the artifact.
```

## Schedule prediction completion count

Original lines 1779–1779.

```tex
All 36 fits and 3,552 predictions (including Query) completed without failure.
```

## Additional editorial wording

The following original wording was replaced on readback; scientific definitions
and values remain in the paper.

### Data-audit inventory wording

```tex
stratum. Realized lengths and target identities are retained in the data audit.
```

### Length-panel acceptance wording

```tex
The actual accepted lengths
```

### Sequence-screening terminology

```tex
is operational sequence screening, not homologous-superfamily isolation.
```

### Study-role table heading

```tex
Setting & Status & Difference & 95\% interval
```

### Internal diagnostic revision title

```tex
\subsection{Reachable versus learned updates: v4 diagnostics}
```

### Internal diagnostic revision name

```tex
The v4 study uses nine paired noise conditions on Observed96.
```

### Internal diagnostic revision caption

```tex
\caption{v4 task-loss diagnostics:
```

### Internal feature-study stage title

```tex
\section{OpenFold Stage A: frozen ESMC feature substitution}
```

### Internal feature-study stage name

```tex
Stage A fixes Train384, Full, 1536 updates,
```

### Internal feature-study stage caption

```tex
these intervals use the Stage A bootstrap stream.
```

### Protenix table completion wording

```tex
Protenix completed factorials:
```

### AtlasFold table completion wording

```tex
AtlasFold completed factorials:
```

### Fresh192 panel acceptance wording

```tex
Accepted panel members are screened against one another as well.
```

### Atlas diagnostic heading

```tex
\section{AtlasFold boundary checks}
```


## Main-text editorial changes

The following exact passages were removed or condensed in the same revision.
Scientific context retained in the revised manuscript can appear alongside
operational wording in these historical excerpts.

### Main-text passage 1

Source: `sections/04_results.tex`.

```tex
recipe from observed evidence before evaluating Fresh192. All 4,800 predictions
completed without failure or new training. The sole primary interaction is
```

### Main-text passage 2

Source: `sections/04_results.tex`.

```tex
Appendix~\ref{app:fresh96}. All 2,400 predictions completed successfully, with
no new training. Factor's Native--Rotated score difference remains positive:
```

### Main-text passage 3

Source: `sections/04_results.tex`.

```tex
replications (Table~\ref{tab:study_roles}). A66 multiplicity and backend limits
```

### Main-text passage 4

Source: `sections/04_results.tex`.

```tex
The completed OpenFold Train384 four-cell controls do not
```

### Main-text passage 5

Source: `sections/04_results.tex`.

```tex
The Protenix Train96/Full follow-up adds three G+ and nine rotated-G+ fits,
matching the training/development manifests and 1536-update budget of the
OpenFold comparison, with Protenix's own fixed recipe.
```

### Main-text passage 6

Source: `sections/04_results.tex`.

```tex
The Protenix Train384 feature study completes both ESM2 and ESMC four cells
```

### Main-text passage 7

Source: `sections/04_results.tex`.

```tex
Historical ESM2 Length48 compares H100 with MI250;
replays give no panel-wide error bound (Appendix~\ref{app:engineering}).
```

### Main-text passage 8

Source: `sections/04_results.tex`.

```tex
Repeating all nine $+C$ fits for 1536 steps gives 864 successful predictions.
The original no-$C$ models remain fixed; the repeat adds no independent seeds
and is not pooled with the original execution. A positive additional
compensation gain for rotated Factor was not re-established. Starting states
were verified identical, and short replays
identified controllable early numerical variability; its contribution to the
full-training outcome difference remains unresolved. Output freedom and
joint optimization remain coupled (Appendix~\ref{app:e2_repeat}).
```

### Main-text passage 9

Source: `sections/04_results.tex`.

```tex
norm for fixed Protenix Train384/Full/1536 models: 24 observed Confirm96-B targets,
720 complete predictions on the same backend.
```

### Main-text passage 10

Source: `sections/03_design.tex`.

```tex
with tangent as the locked main comparison and Full as a secondary comparison.
```

### Main-text passage 11

Source: `sections/03_design.tex`.

```tex
G+ also has three native and nine rotated fits in each completed four-cell setting.
```

### Main-text passage 12

Source: `sections/03_design.tex`.

```tex
training range of 152--383. The original direction and length studies locked new targets before scoring;
```

### Main-text passage 13

Source: `sections/03_design.tex`.

```tex
separates panel novelty from each four-cell study's endpoint role. Fresh192
locks 48 targets per short-chain stratum (actual 128--383 residues) for fixed
Protenix/ESMC models, after refreshing project exposure exclusions; its only
```

### Main-text passage 14

Source: `sections/03_design.tex`.

```tex
\caption{Identity of the complete four-cell studies.
```

### Main-text passage 15

Source: `sections/08_statements.tex`.

```tex
AI assistants were used to assist code development, experiment orchestration,
evidence auditing, and manuscript drafting. Numerical claims are
linked to saved experimental artifacts through a source-hashed generation script;
they are not generated experimental observations. The authors remain responsible
for the content and verification.
```

### Main-text passage 16

Source: `sections/08_statements.tex`.

```tex
The accompanying analysis artifact reconstructs manuscript tables and figures
from fixed score records and includes numerical source mappings and operator
checks. A CPU example executes the sealed compensation writer and optimizer on
synthetic OPM inputs, including zero initialization and joint gradients
(Appendix~\ref{app:e2_compensation}). A one-target OpenFold entry has been executed in a fresh isolated Python
environment: FASTA to recomputed ESM2 features, released adapter, full-atom CIF
and separate fixed-correspondence pair-lDDT scoring (Appendix~\ref{app:single_repro}).
This is a bounded prediction replay. Separately, all nine compensated models
completed a full-budget retraining repeat and coordinate-scoring audit
(Appendix~\ref{app:e2_repeat}). That execution reused the environment, feature
caches and fixed original no-$C$ baselines; it is not a fresh-environment
end-to-end reproduction. Appendix~\ref{app:recipes}
describes the backbone-specific recipes, Appendix~\ref{app:chronology} the study
sequence, and Appendix~\ref{app:scoring} the metrics and pairing. This compact
artifact does not bundle pretrained model weights, the original full structure
data, or an end-to-end retraining environment.

```

## Internal study names

The manuscript now uses descriptive names for the Protenix/AtlasFold feature
study previously called A66. Historical filenames and numerical source keys
retain their original identifiers; the study and its statistical roles are unchanged.
