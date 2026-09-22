"""MSA-module replacement that preserves Mini's pretrained pair stack."""

from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor, nn


class OPMProvider(Protocol):
    """Produce an OPM-shaped update without reading homolog MSA features."""

    def __call__(self, z: Tensor, s_inputs: Tensor) -> Tensor: ...


class StaticOPMProvider(nn.Module):
    """Replay one cached OPM update on every recycle."""

    def __init__(self, update: Tensor) -> None:
        super().__init__()
        if update.ndim < 3:
            raise ValueError("cached OPM update must have shape [..., L, L, C_z]")
        self.register_buffer("update", update.detach().clone(), persistent=True)

    def forward(self, z: Tensor, s_inputs: Tensor) -> Tensor:
        del s_inputs
        update = self.update.to(device=z.device, dtype=z.dtype)
        if update.shape[-3:] != z.shape[-3:]:
            raise ValueError(
                "cached OPM update and pair state disagree: "
                f"{tuple(update.shape)} versus {tuple(z.shape)}"
            )
        return update


class InjectedMSAModule(nn.Module):
    """Bypass MSA featurization/OPM while retaining the final MSA pair stack."""

    def __init__(self, pair_stack: nn.Module, provider: nn.Module) -> None:
        super().__init__()
        self.pair_stack = pair_stack
        self.provider = provider
        self.n_blocks = 1

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: Tensor,
        s_inputs: Tensor,
        pair_mask: Tensor | None,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> Tensor:
        del input_feature_dict
        update = self.provider(z, s_inputs)
        if inplace_safe:
            z += update
        else:
            z = z + update
        _, z = self.pair_stack(
            s=None,
            z=z,
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return z


def make_replay_module(msa_module: nn.Module, update: Tensor) -> InjectedMSAModule:
    """Construct a cached replay module from a one-block Protenix MSA module."""

    try:
        blocks = msa_module.blocks
    except AttributeError as exc:
        raise ValueError("msa_module does not expose blocks") from exc
    if len(blocks) != 1:
        raise ValueError(f"cached replay requires one MSA block, found {len(blocks)}")
    block = blocks[0]
    if not getattr(block, "is_last_block", False):
        raise ValueError("cached replay requires the sole block to be final")
    return InjectedMSAModule(block.pair_stack, StaticOPMProvider(update))
