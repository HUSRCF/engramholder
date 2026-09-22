"""Shared query construction and native task loss for interface controls."""

from __future__ import annotations

import copy

import torch

from engramfold.protenix import locate_single_opm
from engramfold.protenix.query_opm import build_query_msa_embedding
from engramfold.protenix.replay import InjectedMSAModule
from engramfold.protenix.task_adaptation import DifferentiableOPMProvider


@torch.no_grad()
def query_context(model, features):
    """Only sequence/chemical-graph inputs; caller must restore original MSA."""
    from protenix.model.protenix import update_input_feature_dict

    features = update_input_feature_dict(copy.deepcopy(features))
    s_inputs = model.input_embedder(features, inplace_safe=False, chunk_size=None)
    embedding = build_query_msa_embedding(model.msa_module, s_inputs, features)
    opm = locate_single_opm(model)
    normalized = opm.layer_norm(embedding)
    return {
        "query_a": opm.linear_1(normalized).squeeze(-3).detach(),
        "query_b": opm.linear_2(normalized).squeeze(-3).detach(),
        "query_update": opm(embedding, inplace_safe=False, chunk_size=None).detach(),
    }


def native_task_loss(
    model, original_msa, features, labels, update, loss_fn, permutation, audit_gradient=False
):
    from protenix.model.generator import sample_diffusion_training

    calls = []
    handle = original_msa.blocks[0].pair_stack.register_forward_pre_hook(lambda *_: calls.append(1))
    model.msa_module = InjectedMSAModule(
        original_msa.blocks[0].pair_stack, DifferentiableOPMProvider(update)
    )
    s_inputs, s, z = model.get_pairformer_output(
        copy.deepcopy(features), N_cycle=4, inplace_safe=False
    )
    handle.remove()
    if len(calls) != 4:
        raise RuntimeError("pair stack path changed")
    _, coordinate, sigma = sample_diffusion_training(
        noise_sampler=model.train_noise_sampler,
        denoise_net=model.diffusion_module,
        label_dict=labels,
        input_feature_dict=features,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        pair_z=None,
        p_lm=None,
        c_l=None,
        N_sample=1,
        diffusion_chunk_size=1,
        use_conditioning=True,
        enable_efficient_fusion=False,
    )
    pred = {"coordinate": coordinate, "noise_level": sigma, "distogram": model.distogram_head(z)}
    pred, _, _, _ = permutation.permute_diffusion_sample_to_match_label(
        features, pred, labels, stage="train"
    )
    loss, metrics = loss_fn(features, pred, labels, mode="train")
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite native task loss")
    denoise_norm = None
    if audit_gradient:
        grad = torch.autograd.grad(loss, coordinate, retain_graph=True)[0]
        denoise_norm = float(
            torch.autograd.grad(coordinate, update, grad_outputs=grad, retain_graph=True)[0].norm()
        )
        if denoise_norm <= 0:
            raise RuntimeError("no native denoising gradient to injected update")
    info = {
        "noise_level": sigma.detach().cpu().tolist(),
        "pair_stack_forward_calls": len(calls),
        "denoising_update_grad": denoise_norm,
        "bond_mask_count": int(features["bond_mask"].sum()),
        "metrics": {k: float(v.detach().mean()) for k, v in metrics.items() if torch.is_tensor(v)},
    }
    return loss, info
