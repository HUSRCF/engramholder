# Native Update Alignment for Frozen Protein Structure Adaptation

## Abstract

How should a small adapter write into a frozen protein structure predictor? We study residual updates constructed around the predictor's native factor-combination operator, and distinguish the value of this constraint from the orientation of its outputs. Trained orthogonal rotation controls preserve factor structure, parameter count and decoder singular values while changing orientation relative to the frozen downstream network. In Protenix-Mini, native first-order updates outperform these controls by 0.05710 Cα pair-lDDT on a separately locked 96-target panel. Fixed full-factor adapters retain a 0.04252 advantage on 48 prospectively locked proteins of 392–750 residues, beyond their adaptation-training range of 152–383 residues. Experiments with AlphaFold2 weights in OpenFold extend the direction effect to a different folding architecture, but reveal limits: the effect is supported on both tested panels after Train96 adaptation and only on the longer-chain panel after Train384 adaptation. An explicit-factor generic head achieves higher longer-chain scores than the native head. Local gradient compatibility does not establish an explanation of trained structural gains. Together, these results identify orientation relative to frozen downstream computation as a consequential, conditional property of constrained adaptation, rather than establishing universal superiority of native factor heads or a practical replacement for strong folding systems.

## 1. Introduction

A frozen structure predictor already contains operators that transform residue
information into pair representations. An adapter can reuse this computation,
or learn a generic pair update from new sequence features. These alternatives
raise two distinct questions: whether native factor composition is a useful
constraint, and whether its output directions are compatible with the fixed
network that consumes them. Neither question is answered by parameter count alone.

We construct zero-initialized residuals around native operators and investigate
these questions with trained controls. Protenix-Mini provides the initial setting:
a residue-wise predictor uses frozen ESM2 features to modify native query factors,
while retaining the frozen decoder and downstream pair processing. A generic
pair head tests the overall parameterization. Fixed orthogonal output rotations
then retain the factor construction and decoder spectrum while changing its
orientation relative to the downstream model. Rotated adapters receive training;
they are not native adapters damaged only at inference.

The evidence has three levels. First, independently locked Protenix comparisons
establish an advantage over the original generic head and a separate direction
effect. Second, fixed adapters remain useful on complete proteins longer than
those used for adaptation. Third, experiments using AlphaFold2 weights in
OpenFold test a substantially different architecture and expose the conditional
nature of the effect. Native orientation is advantageous in some AF2 settings,
but native factors do not consistently outperform a generic head given explicit
factor access. We report these boundaries alongside the positive results.

Our contribution is an operator-centered adaptation construction and controlled
empirical evidence about its orientation, scope and competitive limits. We do
not introduce OPM itself, claim recovery of evolutionary covariance, or identify
a complete mechanism for folding generalization.

### Related work

PLM-based folding includes HelixFold-Single, ESMFold and the ESM-conditioned
Protenix-Mini route [1,5,9]. Our question is the parameterization of an internal
update in a frozen predictor. Adapters and Side-Tuning establish learning small
modules around pretrained networks [6,7], while ReZero studies residual
initialization [8]. PiSSA, VeRA and ReFT respectively examine pretrained weight
directions, fixed random adaptation matrices and representation interventions
[11–13]. Native directions or activation adaptation are therefore not new in
general. We study input-dependent native factor anchors and trained,
spectrum-preserving direction interventions in structure prediction. OpenFold
provides a trainable AF2 implementation [14]; ColabFold is an additional AF2
execution pipeline [15], not a second independent architecture in our study.

## 2. Native residuals and trained direction controls

### A common construction, with backbone-specific operators

Let $k$ denote an injection site and its invocation, including recycle state.
At this invocation, let $a_k,b_k$ be native factors, $U_{k,0}$ the unmodified
update, and $G_k$ the frozen factor-combination map used to construct a residual.
A sequence-conditioned adapter produces residue increments $(\delta a,\delta b)$:

$$
\Delta U_k=G_k(a_k+\delta a,b_k+\delta b)-G_k(a_k,b_k),
\qquad
U_k^{(R)}=U_{k,0}+\mathcal R_k\Delta U_k.
\tag{1}
$$

$\mathcal R_k$ applies a fixed orthogonal matrix $R_k$ to each pair's output
channels; the native arm uses $R_k=I$. The chosen matrix is reused across
invocations at the selected site. Only the final increment projection is
zero-initialized, making every arm exactly equal to its backbone's baseline
function at initialization. Upstream adapter layers are normally initialized.
Backbone, decoder and PLM parameters remain frozen.

$U_{k,0}$ means the unmodified update **at the current state**, not necessarily
one tensor cached for the whole forward. In AF2 and AtlasFold, live factors
can change with recycle state, including the effects of earlier adapter calls.
The Protenix experiment uses its cached query-only factors. These are different
instantiations of Eq. (1); the shared principle does not require a shared
normalization constant or structure loss.

For Protenix, the implemented residual decoder is
$G_D(a,b)=[D W\operatorname{vec}(a\otimes b)+b_{\rm out}]/(D+0.001)$,
with $D=508.5$, the training-set median MSA depth. The baseline $U_q$ retains
its exact depth-one affine and normalization. $D$ is a scalar in the residual
construction, not homolog information supplied at inference. For OpenFold,
$G_k$ uses the selected layer's native masked OPM and pair-specific denominator;
the Protenix scalar is not transferred.

For the non-OPM AtlasFold extension [16], the frozen map combines projected factors
as $G_k(a_i,b_j)=W_k[a_i-b_j;\,a_i\odot b_j]+b_k$.
Its native AtlasLM remains present. This extension has a fixed experimental
slot below, but its incomplete adapter results are not evidence in this draft.

**Table 1. Executed interfaces and training contracts.** N/R denotes native
and rotated factor heads. All adapters additionally use frozen ESM2-35M
layer-12 features (480 channels); native backbone inputs are retained as stated.

| Item | Protenix-Mini / Tiny | AF2 through OpenFold | AtlasFold (adapter results pending) |
|---|---|---|---|
| Injection | Single MSA-module OPM boundary; downstream pair stack retained | First main Evoformer block's OPM; all 48 blocks retained | First LM-stack block's product/difference update |
| Anchor | Cached query-only factors, dimension 32 | Live factors after current MSA transformations, dimension 32 | Live normalized/projected sequence factors, dimension 128 |
| Residual normalization | $D/(D+0.001)$; exact depth-one baseline retained | Native masked row-count denominator plus layer epsilon; baseline affine retained | Native output projection; no OPM depth normalization |
| Native PLM in adapted backbone | None in default checkpoints; separate official ESM route is a reference | None in AF2; query-only MSA features | AtlasLM-3B retained, plus adapter ESM2-35M |
| Trainable parameters | N/R: 373,824; Generic: 378,272; G+: 378,144 | N/R: 373,824; G+: 378,144 | N/R: 423,168; G+: 415,008 |
| Structure loss | $4L_{\rm MSE}+4L_{\rm bond}+4L_{\rm smooth}+0.03L_{\rm dist}$ | $L_{\rm FAPE}+0.3L_{\rm dist}+L_{\rm torsion}$ | $0.4L_{\rm dist}+2(L_{\rm weighted\ MSE}+L_{\rm smooth})$ |
| Inference | 4 recycles, 5 diffusion steps, 1 sample, seed 101 | 4 trunk passes, structure module, no early stop; seed 20260921 | 5 trunk passes (`num_recycles=4`); 20 diffusion steps at $L\le512$, 30 at $513\le L\le1024$; 1 sample, seed 1 |

The Protenix bond term is zero for the recorded empty protein-only bond masks.
Confidence losses are disabled; OpenFold also excludes masked-MSA loss.
AtlasFold retains native MLM conditioning with probability 0.15. Its pinned
implementation executes `num_recycles+1` trunk passes. Losses retain native
component definitions and supervision masks;
experimental structures enter labels and scoring, not sequence-only inputs.
The appendix specifies precision, optimizer settings and invocation semantics.

### What the intervention controls

At a common factor input and writer parameter state,

$$
\|\mathcal R\Delta U\|_F=\|\Delta U\|_F,
\quad \sigma_j(RW)=\sigma_j(W),
\quad J_R^\top J_R=J^\top J.
\tag{2}
$$

Here $J$ is the local residual writer Jacobian, holding its incoming state
fixed. The downstream responses $J_{\rm down}J$ and
$J_{\rm down}\mathcal R J$ can differ. Independently trained trajectories,
state-dependent anchors and the full recurrent network are not asserted to
have identical Jacobian spectra or optimization geometry. The fixed baseline
and downstream weights are never rotated.

For OPM, the full residual contains a first-order term
$T=\mathcal W(\delta a\otimes b+a\otimes\delta b)$ and quadratic increment
term $Q=\mathcal W(\delta a\otimes\delta b)$. The first-order arm retains
$T$ only. In the Protenix setting without an additional pair-dependent gate,
each output-channel residual has rank at most $2r=64$; this bound need not
survive downstream nonlinear processing. The quadratic term is a combination
of adapter increments, not measured evolutionary covariance.

Generic predicts a pair residual using projected sequence features and clipped
relative sequence separation. G+ additionally receives the relevant native
factors. All heads share parameters across residue positions. G+ changes the
pair MLP width to approximately match parameter count; it is a competitive
explicit-access control, not a pure one-variable ablation. The N/R comparison
asks about orientation within a fixed family; Native−G+ asks which family
performs better. These contrasts need not have the same sign.

## 3. Evaluation design

Confirm96-A is the prospectively locked original Factor−Generic panel;
Confirm96-B is the separately locked primary direction panel. Each contains
96 proteins in four prespecified length strata. Later Protenix and all AF2
comparisons reuse observed Confirm96-B. Length48 was prospectively locked for
fixed Protenix adapters, but was already observed when used for AF2 and later
models. Changing the backbone does not restore blinded target status.

Adaptation expands from Train24 to nested Train96 and Train384; Dev8 remains
the development set. Protenix Train384 contains chains of 152–383 residues.
Length48 selects 16 chains per bin 385–512, 513–640 and 641–768, with actual
lengths 392–750. Inputs are complete sequences. Common observed-coordinate
masks affect scoring, not input length. The panels use single-model experimental
structures, X-ray resolution at most 2.5 Å and at least 90% Cα coverage.
Sequence exclusion rules and their pre-scoring revisions are preserved in
Appendix B and the Length48 supplement. Four known CATH family pairs remain
retrieval-stage misses: sequence screening does not establish family isolation
or foundation-model pretraining exclusion.

The primary score is Cα pair-lDDT: fixed reference distances below 15 Å,
positive distances only, no self-pairs, and strict 0.5/1/2/4 Å error thresholds
[2,3,10]. Missing predictions receive zero credit under the common reference
mask. Residue-averaged Cα-lDDT and fixed-correspondence TM-score normalized by
complete query length are secondary audits of the same structures [4].
No prediction is selected by true structural quality.

For each target $i$, direction effects average rotations within paired training
seed and seeds within target:

$$
d_i=\frac13\sum_{s=1}^3\left(S_{N,i,s}-\frac13\sum_{r=1}^3S_{R,i,s,r}\right).
\tag{3}
$$

We report target means and 20,000-draw paired target-bootstrap intervals,
conditional on fitted models. Seeds, rotations and noise conditions are not
independent proteins. Secondary and post-result intervals are unadjusted for
multiple comparisons. The original panels retain their locked primary
comparisons and continuation rules; follow-ups do not redefine them.

For OpenFold, official AF2 `model_3_ptm` weights remain frozen, with no templates
or homolog rows. Train96 and Train384 each use 15 fits: three native, nine
rotated and three G+. All run 1,536 updates and use the final checkpoint. A
separate development seed and symmetric 384-step calibration over native,
one fixed rotation and G+ select one common learning rate from
$\{5\times10^{-5},10^{-4}\}$; $10^{-4}$ is selected for Train96 and retained
for Train384. AdamW weight decay is 0.01. Protenix direction studies instead
use $5\times10^{-5}$ and their locked native loss. Equal update counts across
architectures do not imply equal computation. ColabFold supplies a separately
executed AF2 query-only reference, not an MSA-enabled comparison.

## 4. Structural evidence in Protenix

On Confirm96-A, task-only Factor scores 0.56241 versus 0.49725 for Generic:
**+0.06516 [0.04936, 0.08288]**, with 77/96 positive target differences and
three positive seed means. The comparison establishes an advantage over this
specific generic parameterization. In a later frozen-checkpoint comparison
on observed Confirm96-B, Factor also exceeds G+ by
+0.05230 [0.03553, 0.06983]. The latter uses independently selected,
head-specific learning rates and is not new blinded confirmation.

**Table 2. Direction interventions in Protenix.** All values are Cα pair-lDDT
native−trained-rotation differences. The first two rows are the primary and
prespecified full-factor secondary comparisons of the locked direction study;
the remaining rows are follow-ups on the already observed panel.

| Backbone / adapter | Adaptation / updates | Confirm96-B difference [95% CI] |
|---|---|---|
| Mini / first-order, primary | Train24 / 384 | +0.05710 [0.04249, 0.07274] |
| Mini / full | Train24 / 384 | +0.04650 [0.03118, 0.06336] |
| Tiny / first-order | Train24 / 384 | +0.02206 [0.01397, 0.03075] |
| Mini / first-order, mean-preserving rotation | Train24 / 384 | +0.10059 [0.07958, 0.12233] |
| Mini / full | Train96 / 1536 | +0.02505 [0.01378, 0.03730] |
| Mini / full | Train384 / 1536 | +0.03951 [0.02800, 0.05216] |

The primary direction comparison has 78/96 positive target differences and
positive means for all seeds and rotation marginals. Supplementary lDDT and
TM-score contrasts support the same direction. Thus factor form, parameter
count and decoder spectrum alone do not determine adaptation quality.
Fixing the channel-mean direction while rotating its orthogonal complement
does not remove the effect. This controls mean-direction mixing, not complete
LayerNorm behavior after adding an unrotated background.

First-order truncation is useful as a control, not a better default method:
first-order−full is −0.01871 at Train24/384 and −0.01764 at Train96/1536,
with intervals below zero. Native expansion from Train96 to Train384 at
1,536 updates gives a smaller gain, +0.00919 [0.00115, 0.01741], with one
seed declining. Its direction interaction is +0.01446 [0.00350, 0.02586]
in a subsequent paired analysis. This changes data composition and per-chain
repetition as well as count, and does not establish a scaling law.

## 5. Architecture and length boundaries

### Fixed Protenix adapters beyond their adaptation lengths

On Length48, native full-factor adapters trained on Train384 score 0.52919,
versus 0.48667 for trained rotations and 0.32122 for query-only. The locked
primary difference is **+0.04252 [0.02899, 0.05765]**, with 41/48 positive
target differences. Residue-averaged lDDT gives +0.04029 and TM-score +0.04782,
both with positive intervals. Native−query (+0.20797) is a descriptive secondary
comparison. All 672 model–target predictions complete without input cropping.

Shared residue updates therefore remain useful beyond the adapter's training
lengths in this test. The experiment does not establish arbitrary-length,
family or pretraining generalization. The three length-bin effects are positive,
but do not establish a monotonic length trend. Protenix G+ was not evaluated
on this longer-chain panel; its AF2 evaluation below is a different experiment.

### AF2: supported effects and competitive boundaries

**Table 3. AF2/OpenFold, fixed 1,536-update endpoints.** Scores average trained
seeds; rotations additionally average three matrices. Both panels were already
observed. G+ remains in the main comparison irrespective of its ranking.

| Adaptation | Panel | Query-only | Native | Rotated | G+ | Native−rotated [95% CI] |
|---|---|---:|---:|---:|---:|---|
| Train96 | Confirm96-B | 0.30191 | 0.50190 | 0.48209 | 0.49534 | +0.01981 [0.00928, 0.03129] |
| Train96 | Length48 | 0.26079 | 0.38872 | 0.37834 | 0.39849 | +0.01038 [0.00403, 0.01705] |
| Train384 | Confirm96-B | 0.30191 | 0.50684 | 0.50381 | 0.51333 | +0.00303 [−0.00675, 0.01218] |
| Train384 | Length48 | 0.26079 | 0.40165 | 0.39040 | 0.40868 | +0.01126 [0.00527, 0.01734] |

All three adapter families improve mean scores over query-only. Direction
effects are supported on both Train96 panels and on Length48 after Train384
adaptation. At Train384 on Confirm96-B, all three metric intervals cross zero;
one of three training seeds favors rotation. This establishes neither a native
advantage nor equivalence in that condition. The longer-chain Train384 effect
has positive intervals for both supplementary metrics, although one TM-score
rotation marginal is slightly negative.

A post-result paired analysis finds Train384 Native−G+ of
−0.00648 [−0.01777, 0.00536] on Confirm96-B and
−0.00703 [−0.01369, −0.00039] on Length48. Train96 Native−G+ on Length48
is also negative, −0.00977 [−0.01674, −0.00288]. These unadjusted intervals
support a longer-chain competitive limitation in the tested recipes. They do
not show that G+ learned a particular alignment mechanism.

The direction-effect change from Train96 to Train384 is
−0.01678 [−0.03214, −0.00261] on Confirm96-B and
+0.00087 [−0.00728, 0.00863] on Length48. This directly contrasts the paired
effects; it does not infer an interaction from separate significance decisions.
The short-panel pattern differs from Protenix. Neither interaction identifies
which architectural or optimization property causes that difference.

### AtlasFold: fixed non-OPM extension slot

**Pending complete matrix; no adapter-effect claim.** This extension asks whether
native orientation matters at a product/difference interface while retaining
AtlasLM-3B. The locked matrix is Train96, 1,536 updates, three native fits,
three rotations crossed with three seeds, and three G+ fits. Symmetric Dev8
calibration has selected the common $10^{-4}$ learning rate. The primary
comparison is native−rotation on observed Confirm96-B; Length48 and G+ are
secondary. Table 1 already specifies the construction independently of results.

The final table will report query-only, native, rotation and G+ scores on both
panels, paired intervals, all seed marginals and prediction failures. A positive,
null or negative outcome fills the same slot and does not change the main
question. Completed original AtlasFold predictions in Table 4 do not count as
this adapter experiment.

## 6. System position

**Table 4. Complete-system references on the same targets.** These are the
executed single-sequence, no-template/no-homolog conditions, not estimates of
each model's best MSA-enabled or multi-sample performance. No cross-system
quality-matched compute budget is claimed.

| Executed system | Confirm96-B | Length48 | Interpretation |
|---|---:|---:|---|
| Protenix-Mini Train384 native adapter | 0.62572 | 0.52919 | Frozen default backbone + ESM2-35M adapter |
| Official Protenix-Mini-ESM | 0.93599 | 0.92139 | Native ESM2-3B single-sequence reference |
| AF2 query-only, OpenFold | 0.30191 | 0.26079 | No native PLM; same AF2 weights as ColabFold row |
| AF2 query-only, ColabFold | 0.30191 | 0.26074 | Pipeline reference, not another architecture |
| AtlasFold, original | 0.95676 | 0.95050 | Native AtlasLM-3B retained; no trained adapter |
| OpenFold3 / OpenBind-0, query-only | 0.36091 | 0.30814 | Pinned OpenBind weights, native Torch path |
| RF3 Benchmark, query-only | 0.41558 | 0.34672 | 10 recycles, 50 sampling steps, one sample |
| ESMFold2 | Pending complete scoring | Pending complete scoring | Full ESMC6B; 20 loops / 100 steps / one sample |

AtlasFold, OpenFold3 and RF3 each complete 144/144 predictions with zero
recorded prediction failures. AtlasFold uses 20/30 length-dependent diffusion
steps and one sample; OpenFold3 uses its pinned prediction preset and one
sample. Kernel paths, weight hashes and remaining cost fields are specified
in the appendix. OpenFold/ColabFold means are close, but this does not prove
coordinate equality. These system rows establish context, not additional
replications of the direction hypothesis.

Official Mini-ESM exceeds the adapted Mini by 0.39220 on Length48. Earlier
matched-hardware measurements found lower allocated memory with the small-PLM
system, but not a quality-matched speedup. Lower parameter count or partial
stage timings do not establish deployment superiority. Newer models' different
pretraining exposures, native PLMs and inference budgets also prevent attributing
cross-system score differences to our adapter construction.

## 7. Mechanism evidence, limits and conclusion

On observed Confirm96-B, an offline oracle finds a modest average native
advantage in local ridge-gradient compatibility at a common query-only state.
Its finite-amplitude loss response is linked to that geometry by construction.
Neither diagnostic establishes a positive target-level association with trained
folding gains. An additional Train8-derived finite parameter displacement does
not establish positive mean transfer or native superiority on Dev8. The earlier
shared differential-response numerical gate remains failed. Full methods,
negative results and precision checks are retained in the appendix; these
experiments do not provide a deployable inference procedure.

A descriptive decomposition locates 75.2% of the Train384 Protenix improvement
in reference pairs separated by at least 24 sequence positions. These pairs
constitute 57.40% of eligible pairs and also show the largest within-group gain
(+0.05148). Different training settings have different score-headroom patterns.
This shows where distance preservation improves, not recovery of coevolution
or a universal long-range specialization.

The central evidence concerns particular checkpoints, interfaces, training
recipes and target distributions. Mean-preserving rotations do not exhaust
all direction controls. Repeatedly used targets and few training seeds limit
inference beyond the fitted systems. Sequence filtering does not guarantee
family or pretraining isolation, and the incomplete historical CATH audit does
not cover all Train384 exposure. The AF2 results explicitly separate direction
sensitivity from the competitive value of the factor constraint. Pending
AtlasFold adaptation and system-only baselines do not strengthen that claim
until their corresponding comparisons are complete.

**Conclusion.** Native operator-centered residuals provide a concrete way to
adapt frozen structure predictors. Trained orthogonal controls show that
orientation relative to frozen downstream computation can affect structural
quality beyond the factor form and decoder spectrum. This effect extends
across tested settings and to AF2, but depends on training and target conditions;
it does not ensure superiority over generic adaptation. Fixed Protenix adapters
also retain useful updates beyond their adaptation-training length range.
These findings define both the value and the limits of native update alignment.

## References

[1] Protenix-Mini: Efficient Structure Predictor via Compact Architecture,
Few-Step Diffusion and Switchable pLM. https://arxiv.org/abs/2507.11839

[2] AlphaFold lDDT implementation, pinned c77e5d2a8961d1a353632c462914ff0a32a950f6.
https://github.com/google-deepmind/alphafold/blob/c77e5d2a8961d1a353632c462914ff0a32a950f6/alphafold/model/lddt.py

[3] OpenStructure 2.12 lDDT documentation.
https://openstructure.org/docs/2.12/mol/alg/lddt/

[4] USalign TMscore, pinned fcb0f9d921415a2095bc509975db7fc1e968af1d.
https://github.com/pylelab/USalign/tree/fcb0f9d921415a2095bc509975db7fc1e968af1d


[5] Fang et al. HelixFold-Single: MSA-free Protein Structure Prediction by Using
Protein Language Model as an Alternative. https://arxiv.org/abs/2207.13921

[6] Houlsby et al. Parameter-Efficient Transfer Learning for NLP. ICML2019.
https://proceedings.mlr.press/v97/houlsby19a.html

[7] Zhang et al. Side-Tuning: A Baseline for Network Adaptation via Additive Side
Networks. https://arxiv.org/abs/1912.13503

[8] Bachlechner et al. ReZero is All You Need: Fast Convergence at Large Depth.
https://arxiv.org/abs/2003.04887

[9] Lin et al. Evolutionary-scale prediction of atomic-level protein structure
with a language model. Science379:1123–1130,2023.
https://doi.org/10.1126/science.ade2574

[10] Mariani, Biasini, Barbato and Schwede. lDDT: a local superposition-free score
for comparing protein structures and models using distance difference tests.
Bioinformatics29:2722–2728,2013. https://doi.org/10.1093/bioinformatics/btt473

[11] Meng et al. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. https://arxiv.org/abs/2404.02948

[12] Kopiczko et al. VeRA: Vector-based Random Matrix Adaptation. https://arxiv.org/abs/2310.11454

[13] Wu et al. ReFT: Representation Finetuning for Language Models. https://arxiv.org/abs/2404.03592

[14] Ahdritz et al. OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization. Nature Methods 21, 1514–1524 (2024). https://www.nature.com/articles/s41592-024-02272-z

[15] Mirdita et al. ColabFold: making protein folding accessible to all. Nature Methods 19, 679–682 (2022). https://www.nature.com/articles/s41592-022-01488-1

[16] AtlasFold official implementation, pinned commit 8ab3aca0e18c8b814d5ca6756b2617a07d72c68d. https://github.com/SeonghwanSeo/atlasfold/tree/8ab3aca0e18c8b814d5ca6756b2617a07d72c68d
