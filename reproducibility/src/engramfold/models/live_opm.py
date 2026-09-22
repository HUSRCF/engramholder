"""Live, single-query OPM residuals for frozen OpenFold and Boltz-1.

The original module produces the baseline, including its own affine and mask
semantics. Only the bias-free difference is constructed here. No anchor is cached
across blocks or recycles, and no Protenix depth normalization is used.
"""
from __future__ import annotations

import torch
from torch import Tensor

from engramfold.models.interface_heads import InterfaceHead
from engramfold.models.native_geometry import orthogonal_rotation


def full_opm_residual(
    a: Tensor, b: Tensor, da: Tensor, db: Tensor, weight: Tensor,
    mask: Tensor, *, normalization: str, eps: float = 1e-3,
    rotation: Tensor | None = None,
) -> Tensor:
    """Compute a one-row masked full residual; shapes [...,L,r], [...,L]."""
    if a.shape != b.shape or da.shape != a.shape or db.shape != b.shape:
        raise ValueError("factor/increment shapes must match")
    if mask.shape != a.shape[:-1] or weight.shape[1] != a.shape[-1] ** 2:
        raise ValueError("mask or output weight shape mismatch")
    if normalization not in {"openfold", "boltz1"}:
        raise ValueError("unknown native normalization")
    mask = mask.to(a)
    a, b, da, db = [x * mask[..., None] for x in (a, b, da, db)]
    w = weight if rotation is None else rotation.to(weight) @ weight
    w = w.reshape(w.shape[0], a.shape[-1], a.shape[-1])
    residual = (torch.einsum("...ir,crs,...js->...ijc", da, w, b + db)
                + torch.einsum("...ir,crs,...js->...ijc", a, w, db))
    count = mask[..., :, None] * mask[..., None, :]
    denominator = count + eps if normalization == "openfold" else count.clamp(min=1)
    return residual / denominator[..., None]


class LiveOPMAdapter(InterfaceHead):
    """Same residue encoder, live query factors, independently trained per arm."""

    def __init__(self, *, backbone: str, kind: str = "factor", rotation_seed=None,
                 factor_dim: int = 32, pair_channels: int = 128, **kwargs):
        if backbone not in {"openfold", "boltz1"}:
            raise ValueError("unsupported backbone")
        if kind not in {"factor", "generic_plus"}:
            raise ValueError("unsupported adapter arm")
        super().__init__(kind, factor_dim=factor_dim, pair_channels=pair_channels, **kwargs)
        self.backbone = backbone
        self.rotate_generic_output = kind == "generic_plus" and rotation_seed is not None
        self.register_buffer("rotation", orthogonal_rotation(pair_channels, rotation_seed))

    def residual(self, features, a, b, weight, mask, eps=1e-3):
        if a.ndim != 2 or features.shape[0] != a.shape[0]:
            raise ValueError("one complete chain per invocation is required")
        hidden = self.hidden(features)
        if self.kind == "factor":
            da, db = self.output_head(hidden).chunk(2, -1)
            return full_opm_residual(a, b, da, db, weight, mask,
                                     normalization=self.backbone, eps=eps,
                                     rotation=self.rotation)
        nodes = self.node(hidden)
        columns = torch.arange(len(a), device=a.device)
        chunks = []
        for start in range(0, len(a), 32):
            rows = torch.arange(start, min(start + 32, len(a)), device=a.device)
            i, j = torch.meshgrid(rows, columns, indexing="ij")
            chunks.append(self._pairs(nodes, i, j, a, b))
        residual = torch.cat(chunks, 0)
        if self.rotate_generic_output:
            # Keep W/b as optimizer parameters in the rotated arm's coordinates.
            # Rotate only the emitted residual, never the query baseline/anchor.
            residual = residual @ self.rotation.to(residual).T
        return residual * (mask[:, None] * mask[None, :])[..., None]


class LiveOPMHook:
    """Removable hook; writer remains outside the frozen model's state dict.

    Recompute factors with the input's graph intact. This is important if the
    selected invocation is later used with differentiable incoming states.
    """

    def __init__(self, module, writer: LiveOPMAdapter, features: Tensor):
        self.writer, self.features = writer, features
        self.calls = []
        self.handle = module.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, baseline):
        m = args[0] if args else kwargs["m"]
        mask = args[1] if len(args) > 1 else kwargs.get("mask")
        if mask is None:
            mask = m.new_ones(m.shape[:-1])
        if m.shape[-3] != 1 or (m.ndim == 4 and m.shape[0] != 1) or m.ndim not in (3, 4):
            raise ValueError("adapter requires a single chain with exactly one MSA row")
        if self.writer.backbone == "openfold":
            ln = module.layer_norm(m)
            a, b = module.linear_1(ln), module.linear_2(ln)
            weight, eps = module.linear_out.weight, module.eps
        else:
            ln = module.norm(m)
            a, b = module.proj_a(ln), module.proj_b(ln)
            weight, eps = module.proj_o.weight, 0.0
        length = m.shape[-2]
        a, b = a.reshape(length, -1), b.reshape(length, -1)
        residual = self.writer.residual(self.features, a, b, weight, mask.reshape(length), eps)
        if baseline.shape[-3:] != residual.shape:
            raise ValueError("baseline/residual shape mismatch")
        self.calls.append(dict(length=length, factor_dim=a.shape[-1],
                               grad_enabled=torch.is_grad_enabled(),
                               pair_channels=baseline.shape[-1]))
        return baseline + residual.reshape_as(baseline)

    def remove(self):
        self.handle.remove()
