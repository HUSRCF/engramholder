# Shared-channel compensation: executable operator example

Run from the artifact root:

```sh
python reproducibility/compensation/smoke.py
python scripts/verify_e2_intervention.py
```

The first command executes the actual sealed E2 writer and optimizer on CPU,
using synthetic features and a small frozen OPM fixture. It needs NumPy and
PyTorch, but no backbone weights, PLM or experimental structure. The second
command reconstructs the original paired structure-score evaluation. These are
two separate checks: the synthetic loss is not evidence about protein quality.

The example covers:

- The original 8,001 skew coordinates and `I + B(exp(A)-I)B.T` construction.
- Identity C and zero residual initialization, including the expected zero C
  gradient before the factor head starts producing a nonzero residual.
- Native and the two selected rotations, with C applied after the completed
  rotated residual. R is already fused into the residual decoder; it is not
  applied twice. The original OPM baseline is left unchanged.
- The exact optimizer function: theta AdamW lr=1e-4/decay=.01; C coordinates
  lr=1e-3/decay=0; betas=(.9,.999), eps=1e-8; one global clip norm of 1.
- Joint gradients, mask preservation, shared-map norm/mean constraints, frozen
  fixture parameters, and removal of the injection hook.

`archived/prospective_orientation.py`, `archived/live_opm.py` and `archived/common.py`
match their sealed execution hashes. The original E2 live-OPM source is isolated
here because the more recent bundled working snapshot also supports rotated G+.
The other three model dependencies are checked against the execution lock.
`rotations.npz` contains the exact sealed arrays, not matrices regenerated from
seeds; the example selects r20270107 and r20270104 from those arrays.
`recipe.json` states the fixed scientific recipe. No optimizer or matrix search
is performed by this example.

## Connecting to an existing OpenFold training environment

The full model needs the pinned OpenFold environment and AF2 model_3_ptm weights,
complete sequence features, ESM2-35M layer12 features, and correctly mapped atom
labels described by the main artifact. The historical training loop used:

```python
writer = ProspectiveOPMAdapter(rotation, learned_channels=True).to(device).eval()
opt = optimizer(writer, channels=True)
model.eval().requires_grad_(False)
# For each locked target and step, with the paired seed + step RNG:
opt.zero_grad(set_to_none=True)
hook = LiveOPMHook(model.evoformer.blocks[0].outer_product_mean, writer, esm_features)
try:
    outputs = model(sequence_features)
    loss, components = native_loss(outputs, labels, config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True)
    opt.step()
finally:
    hook.remove()
```

Here `native_loss` is the FAPE + .3 distogram + supervised-chi loss in the
OpenFold runtime source. The fixed runtime retains four trunk passes, gradients
through the last pass, checkpoint recomputation, and unchanged frozen weights.
The hook must remain installed until backward completes. This snippet documents
the actual integration boundary; it is not a data-preparation or cluster launcher.

One C is shared across all targets, pairs and recycle calls of each fitted model.
The formal experiment independently trained Native+C and each Rotated+C from
zero-residual starts. Assigning C=R.T to an existing checkpoint is not that
experiment. Joint training may change the residual norm even though each C
preserves the norm of its own fixed input.

For evaluation, collect all paired scores before reporting B_N, B_R and T_C.
Average seeds and the two rotations within each target, then resample whole
targets synchronously. The common Native control is counted once. See
`verify_e2_intervention.py`; its original 21 contrasts, two individual rotations,
and three metrics are preserved. Positive target counts and medians are explicitly
post-hoc descriptions, not new primary tests.

This package does not bundle the nine trained E2 checkpoints, complete training
labels, or a tested clean-environment full E2 training launcher. It reproduces
operator execution and score-array analysis, not the full structure-quality result.
