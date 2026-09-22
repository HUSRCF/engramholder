"""Cached final-recycle, paired fixed-branch/dynamic native loss diagnostics.

The original v1/v2 runtime and its failed gate are deliberately unchanged.
"""

import copy

import torch
from torch import nn

from engramfold.experiments.gradient_runtime import LastCycleProvider, rng, restore
from engramfold.models.native_geometry import tensor_sha256
from engramfold.protenix.replay import InjectedMSAModule


def detached(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detached(v) for v in value)
    if isinstance(value, list):
        return [detached(v) for v in value]
    return copy.deepcopy(value)


class ConstantProvider(nn.Module):
    def __init__(self, update):
        super().__init__()
        self.update = update

    def forward(self, z, s_inputs):
        return self.update


class DualLossDiagnostic:
    def __init__(self, model, data, loss_fn, permutation):
        self.model, self.data = model, data
        self.loss_fn, self.permutation = loss_fn, permutation
        self.original = model.msa_module
        self.query = data["query"]["query_update"]
        self.start_rng = rng()
        self.conditions, self.cache = {}, {}

    def full_trunk(self, update, *, capture=False):
        restore(self.start_rng)
        provider = LastCycleProvider(self.query, update)
        injected = InjectedMSAModule(self.original.blocks[0].pair_stack, provider)
        calls = {"msa": 0, "pair": 0}

        def save_msa(module, args, kwargs):
            calls["msa"] += 1
            if capture and calls["msa"] == 4:
                self.cache["msa_args"] = detached(args)
                self.cache["msa_kwargs"] = detached(kwargs)
                self.cache["msa_rng"] = rng()

        def save_pair(module, args, kwargs):
            calls["pair"] += 1
            if capture and calls["pair"] == 4:
                # s is independent of the last-cycle OPM intervention; z is rebuilt.
                self.cache["pair_s"] = detached(args[0])
                self.cache["pair_kwargs"] = detached(kwargs)
                self.cache["pair_rng"] = rng()

        hooks = [injected.register_forward_pre_hook(save_msa, with_kwargs=True),
                 self.model.pairformer_stack.register_forward_pre_hook(save_pair, with_kwargs=True)]
        self.model.msa_module = injected
        try:
            result = self.model.get_pairformer_output(
                copy.deepcopy(self.data["features"]), N_cycle=4,
                inplace_safe=False, mc_dropout=False,
            )
        finally:
            self.model.msa_module = self.original
            for hook in hooks:
                hook.remove()
        if calls != {"msa": 4, "pair": 4}:
            raise RuntimeError(f"wrong recycle count: {calls}")
        if capture:
            self.cache["s_inputs"] = detached(result[0])
            self.cache["pre_injection_hashes"] = provider.calls
        elif provider.calls != self.cache["pre_injection_hashes"]:
            raise RuntimeError("pre-intervention states differ")
        return result

    def replay_trunk(self, update):
        restore(self.cache["msa_rng"])
        injected = InjectedMSAModule(self.original.blocks[0].pair_stack, ConstantProvider(update))
        z = injected(*self.cache["msa_args"], **self.cache["msa_kwargs"])
        restore(self.cache["pair_rng"])
        s, z = self.model.pairformer_stack(self.cache["pair_s"], z, **self.cache["pair_kwargs"])
        return self.cache["s_inputs"], s, z

    def evaluate(self, update, *, capture=False, full=False, components=False):
        from protenix.model.generator import sample_diffusion_training

        sin, s, z = self.full_trunk(update, capture=capture) if capture or full else self.replay_trunk(update)
        features, labels = copy.deepcopy(self.data["features"]), copy.deepcopy(self.data["labels"])
        if capture:
            def denoise(**kwargs):
                self.conditions["noisy"] = detached(kwargs["x_noisy"])
                self.conditions["sigma"] = detached(kwargs["t_hat_noise_level"])
                self.conditions["denoise_rng"] = rng()
                return self.model.diffusion_module(**kwargs)

            augmented, xyz, sigma = sample_diffusion_training(
                noise_sampler=self.model.train_noise_sampler, denoise_net=denoise,
                label_dict=labels, input_feature_dict=features, s_inputs=sin,
                s_trunk=s, z_trunk=z, pair_z=None, p_lm=None, c_l=None,
                N_sample=1, diffusion_chunk_size=1, use_conditioning=True,
                enable_efficient_fusion=False,
            )
            self.conditions["augmented"] = detached(augmented)
        else:
            restore(self.conditions["denoise_rng"])
            sigma = self.conditions["sigma"]
            xyz = self.model.diffusion_module(
                x_noisy=self.conditions["noisy"], t_hat_noise_level=sigma,
                input_feature_dict=features, s_inputs=sin, s_trunk=s, z_trunk=z,
                pair_z=None, p_lm=None, c_l=None, use_conditioning=True,
                enable_efficient_fusion=False,
            )
        prediction = {"coordinate": xyz, "noise_level": sigma,
                      "distogram": self.model.distogram_head(z)}
        dynamic, _, indices, label_indices = self.permutation.permute_diffusion_sample_to_match_label(
            features, dict(prediction), labels, stage="train",
        )
        if label_indices:
            raise RuntimeError("unexpected label permutation")
        if not indices:
            indices = [torch.arange(xyz.shape[-2], device=xyz.device)]
        if capture:
            self.conditions["indices"] = detached(indices)
            self.conditions["loss_rng"] = rng()
        fixed_indices = self.conditions["indices"]
        changed = len(indices) != len(fixed_indices) or any(
            not torch.equal(a, b) for a, b in zip(indices, fixed_indices)
        )
        fixed = dict(prediction)
        fixed["coordinate"] = torch.stack([xyz[i, idx] for i, idx in enumerate(fixed_indices)])
        restore(self.conditions["loss_rng"])
        fixed_loss, fixed_metrics = self.loss_fn(features, fixed, copy.deepcopy(labels), mode="train")
        self.component_losses = {}
        if components:
            # The native metrics are detached; obtain a differentiable component
            # through the same native distogram loss implementation.
            dist = self.loss_fn.distogram_loss(
                logits=prediction["distogram"], true_coordinate=labels["coordinate"],
                coordinate_mask=labels["coordinate_mask"],
                rep_atom_mask=features["distogram_rep_atom_mask"],
            )
            if isinstance(dist, tuple):
                dist = dist[0]
            dist = self.loss_fn.loss_weight["distogram_loss"] * dist
            self.component_losses = {"distogram": dist, "denoising": fixed_loss - dist}
        # Reuse the same graph only when the permutation is identical. Rigid alignment
        # remains the native loss implementation, recomputed on every perturbed call.
        if changed:
            restore(self.conditions["loss_rng"])
            dynamic_loss, dynamic_metrics = self.loss_fn(features, dynamic, copy.deepcopy(labels), mode="train")
        else:
            dynamic_loss, dynamic_metrics = fixed_loss, fixed_metrics
        if not torch.isfinite(fixed_loss) or not torch.isfinite(dynamic_loss):
            raise RuntimeError("nonfinite native loss")
        details = {
            "fixed_loss": float(fixed_loss.detach()), "dynamic_loss": float(dynamic_loss.detach()),
            "dynamic_permutation_changed": changed, "sigma": sigma.detach().cpu().tolist(),
            "coordinate_sha256": tensor_sha256(xyz), "pair_sha256": tensor_sha256(z),
            "fixed_metrics": {k: float(v.detach().mean()) for k, v in fixed_metrics.items() if torch.is_tensor(v)},
            "dynamic_metrics": {k: float(v.detach().mean()) for k, v in dynamic_metrics.items() if torch.is_tensor(v)},
        }
        return fixed_loss, dynamic_loss, fixed_metrics, details
