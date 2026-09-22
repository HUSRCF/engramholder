"""Sequence-only AF2 features and separate native structure labels/losses.

Optional OpenFold imports stay inside functions so Protenix environments are
unchanged. NPZ parameter import uses NumPy; this path does not execute JAX.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def runtime_config(*, activation_checkpointing=False):
    from openfold.config import model_config
    config = model_config("model_3_ptm")
    config.data.common.max_recycling_iters = 3  # four complete trunk evaluations
    config.data.common.resample_msa_in_recycling = False
    config.data.predict.max_msa_clusters = 1
    config.data.predict.max_extra_msa = 1  # one zero-mask padding row; no homolog
    config.data.predict.max_templates = 0
    config.data.predict.masked_msa_replace_fraction = 0.0
    config.model.recycle_early_stop_tolerance = -1
    config.globals.chunk_size = 64
    config.globals.blocks_per_ckpt = 1 if activation_checkpointing else None
    config.model.evoformer_stack.tune_chunk_size = False
    config.model.extra_msa.extra_msa_stack.tune_chunk_size = False
    return config


def enable_nonreentrant_checkpointing():
    """Frozen block inputs need nonreentrant checkpointing for an injected writer.

    Upstream reentrant checkpointing can discard a graph when none of its explicit
    inputs require gradients, even though a hook uses trainable parameters.
    Change only the checkpoint backend, without editing vendored model operations.
    """
    from functools import partial
    import torch.utils.checkpoint
    from openfold.utils import checkpointing
    checkpointing.get_checkpoint_fn = lambda: partial(
        torch.utils.checkpoint.checkpoint, use_reentrant=False,
    )


def sequence_features(sequence, config):
    from openfold.np import residue_constants as rc
    from openfold.data.feature_pipeline import np_example_to_features
    length = len(sequence)
    raw = dict(
        aatype=rc.sequence_to_onehot(sequence, rc.restype_order_with_x, True),
        between_segment_residues=np.zeros(length, dtype=np.int32),
        residue_index=np.arange(length, dtype=np.int32),
        seq_length=np.full(length, length, dtype=np.int32),
        msa=np.array([[rc.HHBLITS_AA_TO_ID[aa] for aa in sequence]], dtype=np.int32),
        deletion_matrix_int=np.zeros((1, length), dtype=np.int32),
        num_alignments=np.ones(length, dtype=np.int32),
    )
    features = np_example_to_features(raw, config.data, mode="predict")
    assert features["aatype"].shape == (length, 4)
    assert features["msa_feat"].shape[:2] == (1, length)
    assert features["extra_msa_mask"].count_nonzero() == 0
    # AF2 MSA features: 23 one-hot channels, deletion flag/value, 23 profile
    # channels and cluster deletion mean; all are query-derived or zero here.
    msa = features["msa_feat"]
    assert msa.shape[-2] == 49
    assert torch.count_nonzero(msa[..., 23:25, :]) == 0
    assert torch.count_nonzero(msa[..., 48, :]) == 0
    # Native summarize_clusters includes 1e-6 in the cluster-count denominator.
    assert torch.equal(msa[..., :23, :] / (1.0 + 1e-6), msa[..., 25:48, :])
    assert not any("template" in k or "all_atom_positions" in k for k in features)
    return features


def structure_labels(path: Path, target, aatype):
    import gemmi
    from openfold.np import residue_constants as rc
    from openfold.data import data_transforms as dt
    from engramfold.evaluation.structure import read_atom_site_positions
    from engramfold.evaluation.structure import _read_cif_block
    length = len(target["sequence"])
    block = _read_cif_block(path)
    entity_sequences = {str(row[0]): gemmi.cif.as_string(row[1]).replace("\n", "").replace(" ", "")
                        for row in block.find(['_entity_poly.entity_id', '_entity_poly.pdbx_seq_one_letter_code_can'])}
    assert entity_sequences[str(target['entity_id'])] == target['sequence']
    _, atoms = read_atom_site_positions(path, label_asym_id=target['source_label_asym_id'])
    positions = torch.zeros(length, 37, 3)
    mask = torch.zeros(length, 37)
    for (index, atom), xyz in atoms.items():
        if atom in rc.atom_order:
            assert 1 <= index <= length
            positions[index - 1, rc.atom_order[atom]] = torch.as_tensor(xyz.copy())
            mask[index - 1, rc.atom_order[atom]] = 1
    labels = dict(aatype=aatype.long(), all_atom_positions=positions, all_atom_mask=mask,
                  seq_mask=torch.ones(length), use_clamped_fape=torch.tensor(1.0))
    for transform in (dt.make_atom14_masks, dt.make_atom14_positions, dt.atom37_to_frames,
                      dt.atom37_to_torsion_angles(''), dt.make_pseudo_beta(''),
                      dt.get_backbone_frames, dt.get_chi_angles):
        labels = transform(labels)
    assert labels['backbone_rigid_mask'].sum() > 0 and labels['chi_mask'].sum() > 0
    return labels


def native_loss(outputs, labels, config):
    from openfold.utils.loss import (
        compute_renamed_ground_truth, distogram_loss, fape_loss, supervised_chi_loss,
    )
    batch = dict(labels)
    batch.update(compute_renamed_ground_truth(batch, outputs['sm']['positions'][-1]))
    terms = dict(
        fape=fape_loss(outputs, batch, config.loss.fape),
        distogram=distogram_loss(logits=outputs['distogram_logits'], **{**batch, **config.loss.distogram}),
        supervised_chi=supervised_chi_loss(outputs['sm']['angles'], outputs['sm']['unnormalized_angles'],
                                          **{**batch, **config.loss.supervised_chi}),
    )
    total = sum(config.loss[name].weight * value for name, value in terms.items())
    return total, terms
