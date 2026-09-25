import types
import torch
from torch.utils.checkpoint import checkpoint
from query_anchor import QueryAnchorHook


class DummyOPM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_out = torch.nn.Linear(1, 1).requires_grad_(False)
        self.eps = .001

    def forward(self, m, mask):
        return m[0, :, :1, None].transpose(-1, -2).expand(2, 2, 1) * 7


def setup():
    module = DummyOPM()
    weight = torch.nn.Parameter(torch.tensor(.2))
    writer = types.SimpleNamespace(backbone='openfold', channels=None)
    writer.residual = lambda f, a, b, w, m, eps: weight * (a[:, None] + b[None, :])
    qs = [dict(a=torch.full((2, 1), float(i + 1)), b=torch.ones(2, 1),
               mask=torch.ones(2), eps=.001) for i in range(4)]
    return module, writer, weight, qs


def test_reference_only_changes_residual_and_backward_reuses_last_anchor():
    module, writer, weight, qs = setup()
    m = torch.ones(1, 2, 1)
    hook = QueryAnchorHook(module, writer, torch.zeros(2, 1), qs)
    try:
        with torch.no_grad():
            for i in range(3):
                y = module(m * (i + 1), torch.ones(1, 2))
                assert torch.allclose(y, torch.full_like(y, 7 * (i + 1) + .2 * (i + 2)))
        y = checkpoint(lambda x: module(x, torch.ones(1, 2)).sin(), m * 4, use_reentrant=False)
        hook.finish_forward()
        y.sum().backward()
        assert torch.allclose(weight.grad, torch.tensor(20.) * torch.cos(torch.tensor(29.)))
        assert [x['anchor_index'] for x in hook.calls] == [0, 1, 2, 3, 3]
    finally:
        hook.remove()


def test_mask_mismatch_fails_instead_of_reusing_reference():
    module, writer, _, qs = setup()
    hook = QueryAnchorHook(module, writer, torch.zeros(2, 1), qs)
    try:
        import pytest
        with pytest.raises(ValueError, match='mask/normalization'):
            module(torch.ones(1, 2, 1), torch.tensor([[1., 0.]]))
    finally:
        hook.remove()
