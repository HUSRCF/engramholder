"""O(L) sequence models for per-residue evolutionary targets."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .pair_predictors import N_BIGRAM_CODES, SequenceEncoder, _bigram_codes


class DenseResiduePredictor(nn.Module):
    """A small convolutional sequence baseline with one output per residue."""

    def __init__(self, *, hidden_dim: int = 128, output_dim: int) -> None:
        super().__init__()
        self.sequence_encoder = SequenceEncoder(hidden_dim=hidden_dim)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.output_head(self.sequence_encoder(tokens))


class FeatureResiduePredictor(DenseResiduePredictor):
    """Same CNN and output head, driven by frozen per-residue PLM features.

    Only a dimensional projection replaces the amino-acid embedding lookup.
    The pretrained model is external, frozen, and absent from this state dict.
    """

    def __init__(self, *, input_dim: int, hidden_dim: int = 128, output_dim: int) -> None:
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim)
        self.input_dim = input_dim
        self.sequence_encoder.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("expected aligned [L, input_dim] frozen residue features")
        encoder = self.sequence_encoder
        hidden = encoder.embedding(features)
        for convolution in encoder.convolutions:
            update = convolution(hidden.transpose(0, 1).unsqueeze(0))
            hidden = hidden + encoder.activation(update.squeeze(0).transpose(0, 1))
        hidden = hidden + encoder.global_projection(hidden.mean(dim=0, keepdim=True))
        return self.output_head(hidden)


class MemoryResiduePredictor(DenseResiduePredictor):
    """Dense baseline augmented with gated bigram-addressed conditional memory."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        output_dim: int,
        memory_dim: int = 32,
        hash_heads: int = 2,
        slots: int = 4096,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, output_dim=output_dim)
        if slots < 2 or memory_dim < 1 or hash_heads < 1:
            raise ValueError("memory dimensions and slots must be positive")
        self.slots = slots
        self.memory_dim = memory_dim
        self.hash_heads = hash_heads
        self.memory_tables = nn.ModuleList(
            nn.Embedding(slots, memory_dim, sparse=True) for _ in range(hash_heads)
        )
        self.memory_projection = nn.Linear(hash_heads * memory_dim, output_dim)
        self.memory_gate = nn.Linear(hidden_dim, output_dim)

    def _indices(self, codes: Tensor, head: int) -> Tensor:
        multipliers = (1_000_003, 1_000_033, 1_000_037, 1_000_081)
        offsets = (97_409, 193_939, 389_117, 778_213)
        return torch.remainder(
            codes * multipliers[head % len(multipliers)]
            + offsets[head % len(offsets)],
            self.slots,
        )

    def forward(self, tokens: Tensor, *, memory_enabled: bool = True) -> Tensor:
        hidden = self.sequence_encoder(tokens)
        base = self.output_head(hidden)
        if not memory_enabled:
            return base
        codes = _bigram_codes(tokens)
        values = torch.cat(
            [
                table(self._indices(codes, head))
                for head, table in enumerate(self.memory_tables)
            ],
            dim=-1,
        )
        memory = self.memory_projection(values)
        gate = torch.sigmoid(self.memory_gate(hidden))
        return base + gate * memory

    def active_raw_keys(self, tokens: Tensor) -> int:
        """Return the number of distinct pre-hash bigram keys in a sequence."""

        return int(torch.unique(_bigram_codes(tokens)).numel())


def maximum_bigram_keys() -> int:
    """Return the finite raw address-space size for this first pilot."""

    return N_BIGRAM_CODES
