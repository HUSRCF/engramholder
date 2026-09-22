"""Small sequence encoders with dense or hashed-memory pair heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
PAD_INDEX = len(AMINO_ACIDS)
DISTANCE_BOUNDARIES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
N_DISTANCE_BUCKETS = 2 * (len(DISTANCE_BOUNDARIES) + 1) + 1
N_BIGRAM_CODES = (len(AMINO_ACIDS) + 1) ** 2


def encode_protein_sequence(sequence: str) -> Tensor:
    """Encode a standard-amino-acid sequence as deterministic integer tokens."""

    try:
        values = [AMINO_ACID_INDEX[amino_acid] for amino_acid in sequence]
    except KeyError as error:
        raise ValueError(f"unsupported amino acid {error.args[0]!r}") from error
    if not values:
        raise ValueError("protein sequence cannot be empty")
    return torch.tensor(values, dtype=torch.long)


def directed_distance_bucket(pair_i: Tensor, pair_j: Tensor) -> Tensor:
    distance = pair_j - pair_i
    boundaries = torch.tensor(DISTANCE_BOUNDARIES, device=distance.device)
    magnitude = torch.bucketize(distance.abs(), boundaries)
    center = len(DISTANCE_BOUNDARIES) + 1
    return center + torch.sign(distance) * (magnitude + 1) * (distance != 0)


def _bigram_codes(tokens: Tensor) -> Tensor:
    following = torch.cat([tokens[1:], tokens.new_tensor([PAD_INDEX])])
    return tokens * (len(AMINO_ACIDS) + 1) + following


def bigram_codes(tokens: Tensor) -> Tensor:
    """Return raw, pre-hash local bigram addresses for every residue."""

    return _bigram_codes(tokens)


def raw_pair_keys(tokens: Tensor, pair_i: Tensor, pair_j: Tensor) -> Tensor:
    """Return collision-free pair keys for the diagnostic key-mean baseline."""

    codes = _bigram_codes(tokens)
    distance = directed_distance_bucket(pair_i, pair_j)
    return (codes[pair_i] * N_BIGRAM_CODES + codes[pair_j]) * N_DISTANCE_BUCKETS + distance


class RawPairKeyMean(nn.Module):
    """Exact training-key mean lookup with a configurable unseen-key fallback."""

    def __init__(self, keys: Tensor, values: Tensor, default: Tensor) -> None:
        super().__init__()
        if keys.ndim != 1 or values.ndim != 2 or values.shape[0] != keys.shape[0]:
            raise ValueError("keys and values must have shapes [K] and [K, C]")
        if default.shape != values.shape[1:]:
            raise ValueError("default must match one lookup value")
        if keys.numel() and not torch.all(keys[1:] > keys[:-1]):
            raise ValueError("keys must be strictly increasing")
        self.register_buffer("keys", keys.long().clone())
        self.register_buffer("values", values.clone())
        self.register_buffer("default", default.clone())

    def forward(self, tokens: Tensor, pair_i: Tensor, pair_j: Tensor) -> Tensor:
        query = raw_pair_keys(tokens, pair_i, pair_j)
        output = self.default.expand(query.shape[0], -1).clone()
        if not self.keys.numel():
            return output
        positions = torch.searchsorted(self.keys, query)
        valid = positions < self.keys.numel()
        safe_positions = positions.clamp(max=self.keys.numel() - 1)
        matched = valid & (self.keys[safe_positions] == query)
        output[matched] = self.values[safe_positions[matched]]
        return output


class SequenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(AMINO_ACIDS) + 1, hidden_dim)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2) for _ in range(layers)
        )
        self.activation = nn.GELU()
        self.global_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 1:
            raise ValueError("SequenceEncoder expects one unbatched token sequence")
        hidden = self.embedding(tokens)
        for convolution in self.convolutions:
            update = convolution(hidden.transpose(0, 1).unsqueeze(0))
            hidden = hidden + self.activation(update.squeeze(0).transpose(0, 1))
        return hidden + self.global_projection(hidden.mean(dim=0, keepdim=True))


class PairFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, distance_dim: int = 32) -> None:
        super().__init__()
        self.distance_embedding = nn.Embedding(N_DISTANCE_BUCKETS, distance_dim)
        self.output_dim = hidden_dim * 3 + distance_dim

    def forward(self, hidden: Tensor, pair_i: Tensor, pair_j: Tensor) -> Tensor:
        left = hidden[pair_i]
        right = hidden[pair_j]
        distance = self.distance_embedding(directed_distance_bucket(pair_i, pair_j))
        return torch.cat([left, right, left * right, distance], dim=-1)


class DensePairAdapter(nn.Module):
    """Continuous query-only baseline operating only on requested residue pairs."""

    def __init__(
        self, hidden_dim: int = 128, distance_dim: int = 32, output_dim: int = 128
    ) -> None:
        super().__init__()
        self.sequence_encoder = SequenceEncoder(hidden_dim=hidden_dim)
        self.pair_encoder = PairFeatureEncoder(hidden_dim=hidden_dim, distance_dim=distance_dim)
        pair_dim = self.pair_encoder.output_dim
        self.pair_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def pair_features(self, tokens: Tensor, pair_i: Tensor, pair_j: Tensor) -> Tensor:
        hidden = self.sequence_encoder(tokens)
        return self.pair_encoder(hidden, pair_i, pair_j)

    def forward(self, tokens: Tensor, pair_i: Tensor, pair_j: Tensor) -> Tensor:
        return self.pair_head(self.pair_features(tokens, pair_i, pair_j))


class _MemoryAdapterBase(DensePairAdapter):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        distance_dim: int = 32,
        output_dim: int = 128,
        slots: int = 2**18,
        memory_dim: int = 32,
        hash_heads: int = 2,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, distance_dim=distance_dim, output_dim=output_dim)
        if slots < 2 or memory_dim < 1 or hash_heads < 1:
            raise ValueError("memory dimensions and slots must be positive")
        self.slots = slots
        self.memory_dim = memory_dim
        self.hash_heads = hash_heads
        self.memory_tables = nn.ModuleList(
            nn.Embedding(slots, memory_dim, sparse=True) for _ in range(hash_heads)
        )
        self.memory_projection = nn.Linear(hash_heads * memory_dim * 2, output_dim)
        self.memory_gate = nn.Linear(self.pair_encoder.output_dim, output_dim)

    def memory_values(
        self, tokens: Tensor, pair_i: Tensor, pair_j: Tensor
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, tokens: Tensor, pair_i: Tensor, pair_j: Tensor) -> Tensor:
        pair_features = self.pair_features(tokens, pair_i, pair_j)
        base = self.pair_head(pair_features)
        left_memory, right_memory = self.memory_values(tokens, pair_i, pair_j)
        memory = self.memory_projection(torch.cat([left_memory, right_memory], dim=-1))
        gate = torch.sigmoid(self.memory_gate(pair_features))
        return base + gate * memory


class SingleSiteMemory(_MemoryAdapterBase):
    """Hashed single-site N-gram memory composed into pair predictions."""

    def _indices(self, codes: Tensor, head: int) -> Tensor:
        multiplier = (1_000_003, 1_000_033, 1_000_037, 1_000_081)[head % 4]
        offset = (97_409, 193_939, 389_117, 778_213)[head % 4]
        return torch.remainder(codes * multiplier + offset, self.slots)

    def memory_values(
        self, tokens: Tensor, pair_i: Tensor, pair_j: Tensor
    ) -> tuple[Tensor, Tensor]:
        codes = _bigram_codes(tokens)
        left = torch.cat(
            [
                table(self._indices(codes[pair_i], head))
                for head, table in enumerate(self.memory_tables)
            ],
            dim=-1,
        )
        right = torch.cat(
            [
                table(self._indices(codes[pair_j], head))
                for head, table in enumerate(self.memory_tables)
            ],
            dim=-1,
        )
        return left, right


class PairAddressedMemory(_MemoryAdapterBase):
    """Two-ended N-gram plus directed-distance hashed conditional memory."""

    def _indices(
        self, left_codes: Tensor, right_codes: Tensor, distance_codes: Tensor, head: int
    ) -> Tensor:
        constants = (
            (1_000_003, 1_000_033, 1_000_037),
            (1_000_081, 1_000_099, 1_000_117),
            (1_000_121, 1_000_133, 1_000_151),
            (1_000_159, 1_000_171, 1_000_183),
        )[head % 4]
        mixed = left_codes * constants[0]
        mixed = torch.bitwise_xor(mixed, right_codes * constants[1])
        mixed = torch.bitwise_xor(mixed, distance_codes * constants[2])
        return torch.remainder(mixed, self.slots)

    def memory_values(
        self, tokens: Tensor, pair_i: Tensor, pair_j: Tensor
    ) -> tuple[Tensor, Tensor]:
        codes = _bigram_codes(tokens)
        distance = directed_distance_bucket(pair_i, pair_j)
        forward_values = []
        reverse_values = []
        for head, table in enumerate(self.memory_tables):
            forward = self._indices(codes[pair_i], codes[pair_j], distance, head)
            reverse = self._indices(codes[pair_j], codes[pair_i], distance, head)
            forward_values.append(table(forward))
            reverse_values.append(table(reverse))
        return torch.cat(forward_values, dim=-1), torch.cat(reverse_values, dim=-1)
