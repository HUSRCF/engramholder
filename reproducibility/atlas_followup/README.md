# AtlasFold: bounded propagation and Native-only calibration

This supplement supports the short Atlas boundary appendix. It contains no new
training and does not test a Factor/G+ rotation interaction.

- `propagation_protocol.md`: two previously observed chains, three Native parents,
  and their C-only/head-only continuations; Query plus nine adapted models per
  chain gives 20 observed instances, not 20 independent proteins.
- `records/<target>/`: original per-pass, same-site statistics, completion/smoke
  receipts, and unrounded C-alpha coordinates for all 20 instances. Each state
  difference uses that target's Query at the same pass and position. Complete
  Query hidden tensors remain in the execution archive and are not bundled.
- `propagation.png` / `propagation.pdf`: original full propagation figure. Shaded
  ranges span three seeds and five passes, not confidence intervals. The plotted
  ratio has a position-dependent Query denominator and is not a transmission
  rate; the manuscript table also reports absolute difference and Query RMS.
- `calibration_protocol.md`: six completed Native-only, 512-update continuations;
  C-only versus head-only is the single primary comparison on observed Confirm96-B.
  C-only versus Query and other contrasts are descriptive, unadjusted comparisons.
- `archived/`: the propagation observer, original aggregation/plotting script and
  calibration scorer. These document implementation identity and are not a
  standalone GPU reproduction environment.

The root `evidence/atlas_propagation_*.json` files retain the execution lock and
full aggregation. `evidence/atlas_posttraining_*.json` provides the calibration
lock, completion receipt, 960 model-target score records and complete three-metric
analysis. The later table does not replace any historical Atlas baseline.

From the artifact root, run:

```sh
python scripts/verify_atlas_followup.py
python scripts/build_paper_assets.py
```

The verifier rebuilds all state-summary medians/ranges, final pair-distance
changes from saved coordinates, and 18 calibration contrasts using the original
20,000-draw target bootstrap. It does not recompute hidden-state tensors or score
coordinates against reference structures. Original and redacted-file identities
are handled by the artifact's provenance map.

State differences persist to the trunk output while final structures change
little in these cases. This does not locate a unique cause, establish information
redundancy, or show that state conditioning or a later hook would improve quality.
