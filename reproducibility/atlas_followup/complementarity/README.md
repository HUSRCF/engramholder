# Fixed Atlas native/ESMC readout study

This is an observed Train96/Dev8 diagnostic, not folding-adapter training or a
new-target test. Nine 131,520-parameter readers each receive 1,536 updates.
The common frozen inputs are native single and attention-derived pair features
after the first-pass LM stack. The matched auxiliary branch receives a fixed
128-dimensional view of native single features, aligned ESMC-600M features, or
the same ESMC features under a fixed within-sequence residue permutation.
All branches have nonzero inputs and identical trainable capacity. This does
not match input conditioning or optimization difficulty.

The sole primary endpoint is final Dev8 cross-entropy for native-view minus
aligned ESMC, positive for improvement. The mean is +0.002506, with the
conditional target interval [-0.000186, +0.005430]; one of three seed means
is negative. Aligned ESMC beats permuted ESMC on all-pair CE, but the close-pair
subset reverses this ordering. All subset comparisons are retained and are
unadjusted secondary diagnostics. This does not establish complete information
redundancy, conditional mutual information, or improved folding quality.

Contents:

- `PROTOCOL.md`: original fixed design, normalization/projection, labels and budget.
- `formal/`: all nine completion receipts, Dev8 scores at the four fixed nodes,
  and final Train96 scores. Only the final node is the reported endpoint.
- `archived/`: the exact locked probe implementation and original analyzer;
  these require the execution repository and caches, and are not a standalone
  GPU environment. Trained reader checkpoints and feature/label tensors are
  retained in the execution archive, not this bundle.
- `smoke/` and `analysis/`: original recovery and remote integrity receipts.
- `engineering_failure_clarification.json`: the cache-padding repair occurred
  before any formal update; the initially failing chain was the shortest
  engineering target, correcting the immutable amendment's wording.
- `esmc_propagation/`: six Native model observations on the same two old chains,
  saved C-alpha coordinates and four engineering replay receipts. No Rotated
  arm or structure-quality scoring was added. One seed was historically trained
  on H100; all these inference observations use the same MI250 environment.

The four `evidence/atlas_probe_*.json` / `atlas_esmc_propagation_summary.json`
snapshots hold the execution lock, summary and secondary statistics. The
propagation retained its pre-repair lock, explicitly bound by the amended probe
lock. Original and redacted hashes are mapped by the anonymous package.

Run `python scripts/verify_atlas_probe.py` from the artifact root. It reconstructs
all six prespecified primary/secondary score contrasts, seed and target means,
the 20,000-draw target bootstrap, and final distance changes from saved
coordinates. Eight targets remain the uncertainty unit after seed aggregation.
Receipt checks are not an independent rerun of training or feature extraction.
