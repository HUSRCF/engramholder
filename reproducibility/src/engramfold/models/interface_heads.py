"""Query-anchored heads for same-boundary, same-U supervision controls.

These are new initializations, not the historical native-factor checkpoints.
The decoder is frozen. Only the predictor parameters are optimized.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .residue_predictors import FeatureResiduePredictor


class FrozenOPMDecoder(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor, *, factor_dim: int, depth: float, eps: float):
        super().__init__()
        if weight.shape != (bias.numel(), factor_dim**2):
            raise ValueError("incompatible OPM decoder dimensions")
        if depth <= 0 or eps < 0:
            raise ValueError("invalid OPM normalization")
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("bias", bias.detach().clone())
        self.factor_dim = factor_dim
        self.depth, self.eps = float(depth), float(eps)

    def forward(self, a: Tensor, b: Tensor, pairs: Tensor | None = None) -> Tensor:
        weight = self.weight.reshape(-1, self.factor_dim, self.factor_dim)
        if pairs is None:
            value = torch.einsum("ic,oce,je->ijo", a, weight, b)
        else:
            value = torch.einsum("pc,oce,pe->po", a[pairs[:, 0]], weight, b[pairs[:, 1]])
        return (self.depth * value + self.bias) / (self.depth + self.eps)


class InterfaceHead(nn.Module):
    """Native factor increments or a nonlinear generic pair residual.

    Both start at exactly U_query. Native-factor updates are centered through the
    fixed depth decoder, so the query's depth=1 affine is not silently replaced.
    Both receive the same query update; only the factor head uses native query
    factors as its parameterization anchor. That inherited coordinate prior is an
    explicit part of the tested parameterization, not extra MSA teacher labels.
    """

    def __init__(
        self,
        kind: str,
        *,
        input_dim: int = 480,
        hidden_dim: int = 128,
        factor_dim: int = 32,
        pair_channels: int = 128,
        pair_latent: int = 32,
        pair_hidden: int | None = None,
    ):
        super().__init__()
        if kind not in {"factor", "generic", "generic_plus"}:
            raise ValueError("unknown interface head")
        if pair_hidden is None:
            pair_hidden = 192 if kind == "generic_plus" else 256
        base = FeatureResiduePredictor(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=2 * factor_dim
        )
        self.encoder = base.sequence_encoder
        self.kind, self.input_dim = kind, input_dim
        if kind == "factor":
            self.output_head = base.output_head
        else:
            self.node = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, pair_latent))
            self.output_head = nn.Sequential(
                nn.Linear(
                    2 * pair_latent + 1 + (2 * factor_dim if kind == "generic_plus" else 0),
                    pair_hidden,
                ),
                nn.GELU(),
                nn.Linear(pair_hidden, pair_channels),
            )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def hidden(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("expected aligned [L,input_dim] PLM features")
        h = self.encoder.embedding(features)
        for convolution in self.encoder.convolutions:
            h = h + self.encoder.activation(convolution(h.T.unsqueeze(0)).squeeze(0).T)
        return h + self.encoder.global_projection(h.mean(0, keepdim=True))

    def forward(
        self,
        features: Tensor,
        *,
        query_update: Tensor,
        query_a: Tensor,
        query_b: Tensor,
        decoder: FrozenOPMDecoder,
        pairs: Tensor | None = None,
        row_chunk: int = 32,
    ) -> Tensor:
        h = self.hidden(features)
        baseline = query_update if pairs is None else query_update[pairs[:, 0], pairs[:, 1]]
        if self.kind == "factor":
            da, db = self.output_head(h).chunk(2, -1)
            residual = decoder(query_a + da, query_b + db, pairs) - decoder(query_a, query_b, pairs)
        else:
            nodes = self.node(h)
            if pairs is not None:
                residual = self._pairs(nodes, pairs[:, 0], pairs[:, 1], query_a, query_b)
            else:
                if row_chunk < 1:
                    raise ValueError("row_chunk must be positive")
                length = len(nodes)
                columns = torch.arange(length, device=nodes.device)
                chunks = []
                for start in range(0, length, row_chunk):
                    rows = torch.arange(start, min(start + row_chunk, length), device=nodes.device)
                    i, j = torch.meshgrid(rows, columns, indexing="ij")
                    chunks.append(self._pairs(nodes, i, j, query_a, query_b))
                residual = torch.cat(chunks, 0)
        return baseline + residual

    def _pairs(
        self, nodes: Tensor, i: Tensor, j: Tensor, query_a: Tensor, query_b: Tensor
    ) -> Tensor:
        separation = (i - j).clamp(-32, 32).to(nodes.dtype).unsqueeze(-1) / 32
        parts = [nodes[i], nodes[j]]
        if self.kind == "generic_plus":
            parts.extend([query_a[i], query_b[j]])
        parts.append(separation)
        return self.output_head(torch.cat(parts, -1))


def common_update_loss(
    prediction: Tensor, target: Tensor, *, channel_scale: Tensor, pair_weight: Tensor
) -> Tensor:
    """Same target, positive train-only scale and pair weights for either head.

    Inputs are sampled [P,C] updates, not factor targets. There is no factor loss.
    Sampling, masks and channel scales must be supplied from one shared manifest.
    """
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("expected matching [P,C] predictions and targets")
    if channel_scale.shape != (prediction.shape[-1],) or pair_weight.shape != prediction.shape[:1]:
        raise ValueError("invalid shared scale/weight shapes")
    if not torch.isfinite(channel_scale).all() or not (channel_scale > 0).all():
        raise ValueError("channel scales must be finite and positive")
    if not torch.isfinite(pair_weight).all() or (pair_weight < 0).any() or pair_weight.sum() <= 0:
        raise ValueError("pair weights must be nonnegative with positive sum")
    error = ((prediction - target) / channel_scale).square().mean(-1)
    return (error * pair_weight).sum() / pair_weight.sum()
