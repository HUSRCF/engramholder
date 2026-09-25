# Post-hoc compensation numerical diagnosis

This is an execution audit, separate from the fixed scientific score evidence. It changes no original model, training recipe, panel score, endpoint, or confidence interval. The manuscript's v11 score-input lock remains unchanged.

## What was checked

`historical_starting_state_audit.json` compares the saved initial checkpoints for all nine original/repeated compensated fits. All 16 parameter tensors and three buffers per pair have equal values and byte hashes; recorded optimizer, Torch CPU/CUDA RNG, learning-rate and data-position states agree, as do complete training schedules. Absolute machine paths are replaced by `EXPERIMENTS_ROOT`; scientific values and original checkpoint hashes are retained. Python and NumPy RNG were not saved in the historical checkpoints.

`compact_summary.json` reports the two numerical conditions and original source hashes. On the same physical H100, two historical-setting replays ran three updates each. The first differences among captured objects occurred in the second backward pass: gradients for the input projection and first two convolution layers differed while captured forward activations, coordinates and loss agreed. The differences subsequently reached parameters and the next loss. The total gradient norm at the second step still matched, illustrating why scalar logs miss early differences.

Two additional replays set `torch.backends.cudnn.deterministic=True`. Both conditions already had `cudnn.benchmark=False` and TF32 disabled; the global deterministic-algorithms flag remained false. The deterministic pair matched through three updates, then each restored its own third-update checkpoint in a new process and continued to eight. Every captured tensor byte hash matched between those two replay paths, including gradients, parameters, optimizer moments and coordinates. These are restored prefixes, not an uninterrupted eight-update reproducibility test. Total diagnostic updates: 22.

## Capture and interpretation boundaries

The executed script is preserved unchanged in `archived/gpu_trace.py`. It requires the original locked training assets, environment and upstream model code; this directory is not a self-contained full retraining environment. Full tensor snapshots remain in the experiment archive; the compact public summary carries their comparisons and provenance hashes.

The script captures adapter-module activations, all trainable parameter gradients, before/after writer state, AdamW tensor state and final coordinates. It does not capture every backbone intermediate, native anchor, residual or adjoint. Detached CPU copies synchronize GPU work and change launch timing. The first captured difference is not proof of the first arithmetic operation that differed in the historical runs.

The checkpoint restoration audit and trace's tensor comparator are supplemented by byte-hash and optimizer parameter-group checks. Python and NumPy ambient RNG states differed across fresh diagnostic processes; Torch/CUDA RNG and actual feature/label hashes matched. Data ordering uses a separately seeded local generator. The published `all_captured_tensors_byte_equal` results do not mean every process state was identical.

The unchanged script's opening claim that it never reads evaluation targets is too broad: `contract` hashes all locked data assets, including historical evaluation assets, for integrity. Diagnostic training uses only Train96; it neither evaluates the panel nor reads panel scores. This clarification preserves the exact executed source instead of rewriting its provenance.

The deterministic cuDNN setting removed observed differences in these paired short paths. It does not establish the unique historical kernel responsible, deterministic full-budget training, or an explanation for the original versus repeated compensation endpoint difference. The original extra-benefit result remains unconfirmed by its same-seed full-training repeat.
