# Protenix Train96 G3 + GR9 — 2026-09-23

Authorized new matched-data-size four-cell completion. Reuse archived Mini Full native3 from native_direction96_20260919 and ordinary rotated9 from native_direction_extension_20260919, each1536 updates. Add G+ native3 and fixed output-rotated G+9; never substitute old Generic or Train384 models.

Executable preparation verifies Train96 hash `17aa6ed1922e1f672a68e6733f915ea7b03bc78e3b8da35a649cd32ca347241c` and Dev8 hash `5a6d6de1394d1e329f39be0b385dca1712e7934b980858b4147740baaad28021`. These match OpenFold. Cache IDs must match the manifests exactly in order.

Frozen Mini default0.5.0, query-only profile/zero deletion, intact pair stack, four recycles; frozen ESM2-35M layer12/480. G+ explicitly receives query factors, free output affine zero initialized. Apply each fixed128-channel orthogonal rotation only to completed pair residual; baseline untouched. Protenix fixed decoder D508.5 comes from original Train24 depth median; do not refit it. G+ does not secretly inherit a different feature or task loss.

Seeds20260923/24/25; rotations20261001/02/03. AdamW lr5e-5 wd1e-4 betas(.9,.999) eps1e-8; global clip1; original Train96 schedule and native structure loss;1536 finalupdates, no new calibration. Same per-step noise seed. FP32. Atomic writer+optimizer resume32steps; HIP-only bounded two retries120seconds, retain failed attempts.

Engineering: hash all original assets and12 existing checkpoints; CPU nonzero native/rotation implementation replay; G+ absorption and paired initialization; first/longestTrain96 two-step structure backprop for G and all3 rotations, frozen backbone, exact zero baseline, nonzero output/encoder gradients; one run resume thirdstep. Release all12 only after checks. Do not choose targets or arms by performance.

Resource queue: independent DiamondHill controller waits for the already-running PLM-A66 pipeline's complete marker (and no live children) before taking GPUs. Preserve current6+2 dynamic schedule and all fits. Then up to8 logical GPUs; no simultaneous independent controllers sharing a card. This is queued preparation, not a claim training has started.

Initial evaluation: new12×oldConfirm96-B=1152 full c4/s5 predictions with seed101 and original frozen scoring. Old observed-panel followup. Do not add Train96 Length48 matrix. Fresh96 cross-backbone evaluation remains a later explicitly locked inference extension, not an implicit N192 experiment. No new targets selected here. Report full four cells, rowwise effects and direct target-paired Psi; preserve all seeds/rotations/failed targets. New result not available at lock time.
