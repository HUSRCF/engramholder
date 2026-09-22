"""Native full/tangent residuals with fixed isometric output interventions."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor

from engramfold.models.interface_heads import FrozenOPMDecoder, InterfaceHead


def orthogonal_rotation(channels: int, seed: int | None) -> Tensor:
    if seed is None:
        return torch.eye(channels, dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(channels, channels, dtype=torch.float64, generator=gen)
    # DiamondHill's ROCm torch build lacks CPU LAPACK. NumPy provides the same
    # CPU float64 QR construction without changing or consuming the torch RNG.
    q, r = np.linalg.qr(matrix.numpy())
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return torch.from_numpy(q * signs[None, :]).float()


def tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def bilinear(a: Tensor, b: Tensor, weight: Tensor, pairs: Tensor | None = None) -> Tensor:
    """Bias-free decoder; weight already includes fixed depth scaling."""
    if pairs is None:
        return torch.einsum("ik,ckl,jl->ijc", a, weight, b)
    return torch.einsum("pk,ckl,pl->pc", a[pairs[:, 0]], weight, b[pairs[:, 1]])


class NativeGeometryHead(InterfaceHead):
    def __init__(self, *, order="full", rotation_seed=None, **kwargs):
        if order not in {"full", "tangent"}:
            raise ValueError("order must be full or tangent")
        super().__init__("factor", **kwargs)
        self.order = order
        self.rotation_seed = rotation_seed
        self.register_buffer(
            "rotation", orthogonal_rotation(kwargs.get("pair_channels", 128), rotation_seed)
        )

    def geometry(self):
        return {
            "order": self.order,
            "rotation_seed": self.rotation_seed,
            "rotation_sha256": tensor_sha256(self.rotation),
        }

    def increments(self, features):
        return self.output_head(self.hidden(features)).chunk(2, -1)

    def effective_weight(self, decoder):
        # Fused channel rotation; no LxLxC matrix multiplication at inference.
        weight = self.rotation.to(decoder.weight) @ decoder.weight
        return weight.reshape(-1, decoder.factor_dim, decoder.factor_dim) * (
            decoder.depth / (decoder.depth + decoder.eps)
        )

    def components(self, features, *, query_a, query_b, decoder, pairs=None):
        da, db = self.increments(features)
        w = self.effective_weight(decoder)
        tangent = bilinear(da, query_b, w, pairs) + bilinear(query_a, db, w, pairs)
        quadratic = bilinear(da, db, w, pairs)
        return tangent, quadratic

    def forward(
        self,
        features,
        *,
        query_update,
        query_a,
        query_b,
        decoder: FrozenOPMDecoder,
        pairs=None,
        row_chunk=32,
    ):
        da, db = self.increments(features)
        weight = self.effective_weight(decoder)
        right = query_b + db if self.order == "full" else query_b
        residual = bilinear(da, right, weight, pairs) + bilinear(query_a, db, weight, pairs)
        baseline = query_update if pairs is None else query_update[pairs[:, 0], pairs[:, 1]]
        return baseline + residual


def load_geometry_writer(task, device="cpu"):
    spec = task["geometry"]
    writer = NativeGeometryHead(order=spec["order"], rotation_seed=spec["rotation_seed"]).to(device)
    writer.load_state_dict(task["writer"])
    if writer.geometry() != spec:
        raise ValueError("rotation/geometry provenance mismatch")
    return writer
