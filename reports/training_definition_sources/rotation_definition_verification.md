# Ordinary rotation definition: source and CPU verification

The ordinary dense rotation sampler was verified against execution-locked source. The checked source is `src/engramfold/models/native_geometry.py`, SHA-256 `3173f554ed252ffbd3f1d5a261e91155a157cc41bcb24b2536c00f6bace0abfe`. This hash is recorded in the Protenix direction, OpenFold rotation, OpenFold ESMC A, historical AtlasFold, Protenix Train96 four-cell, and cross-backbone A66 source locks.

The sampler creates a dedicated CPU PyTorch generator from the rotation seed, draws a 128-by-128 standard-normal matrix in float64, computes NumPy QR, multiplies each Q column by the sign of the corresponding triangular diagonal (zero uses +1), and converts the result to float32. It does not constrain the determinant to +1. The exact ideal-arithmetic group is therefore O(128), allowing reflections. The native control uses the identity matrix.

The matrix is registered as a non-trainable buffer. Factor applies it by premultiplying the frozen bias-free output weight; G+ applies it to the emitted residual. The column-vector convention is Rδ, and row-stored tensors use δRᵀ. The query baseline is unchanged. A writer uses the same matrix for all pairs, targets, steps and invocations of the adapted interface. At use sites the stored matrix follows the decoder/residual device and dtype, so float32 storage does not imply all autocast operations are float32.

Ordinary rotation seeds 20261001–20261003 and common formal training seeds 20260923–20260925 are separate. Because the rotation generator is local, rotation construction does not consume the global torch random stream or the NumPy random stream. Within a fixed architecture and training seed, changing only the rotation seed does not change the initialized trainable weights. This statement does not imply matched gradients or optimization trajectories after nonzero residuals develop.

A CPU-only probe extracted the exact sampler function and used torch `2.12.0a0+git78258b9`, NumPy `2.5.2`, one CPU thread, and empty CUDA/HIP device visibility. It launched no training or prediction. For all three seeds, generated float32 matrix hashes matched the historical Protenix rotation records exactly:

| Seed | SHA-256 of contiguous float32 matrix bytes | Determinant |
|---|---|---:|
| 20261001 | `ad5431b5f729517f7a1226cb00f2f93d0b52fe5d3d9a57cd4f3dcab9a60d08df` | +0.9999999795 |
| 20261002 | `354c43014e7e1e27eef7c4de0994c0b6e62bbe76640a5bba19d1964eb2c84fd8` | −0.9999999881 |
| 20261003 | `5579c9af856a23f6b43989634cac3a814d6180086042903d9d3507eb6945a0b3` | +0.9999999925 |

The maximum absolute entry of RᵀR−I was below 2.82e−8. Global torch and NumPy RNG states were unchanged, and reseeding the global torch RNG produced the same rotation. The negative determinant of the second actual matrix directly confirms that the ordinary family is not restricted to SO(128). Probe code and machine-readable output accompany this note as `rotation_cpu_probe.py` and `rotation_cpu_probe.json`.

This construction is separate from the historical mean-preserving controls, the prospective conjugate family, the signed permutations, and the learned compensator. In particular, the prospective family's `special_orthogonal` helper uses NumPy's own Gaussian generator and explicitly corrects the determinant; the ordinary sampler does neither. The learned compensator uses identity-initialized trainable skew coordinates and a matrix exponential rather than a sampled ordinary rotation.

The verification establishes source agreement and the historical matrix bytes. It does not assert bitwise equivalence for all future library versions or audit every training checkpoint or optimizer state.
