"""Run the actual E2 writer and optimizer on synthetic OPM inputs, on CPU.

This is an integration/gradient check, not a protein-quality or training-result
reproduction. The frozen OPM fixture has the real hook's API; no complete
Evoformer, structure module, PLM or experimental labels are loaded.
"""
from pathlib import Path
import copy
import hashlib
import json
import platform
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'src'))
from archived.prospective_orientation import ProspectiveOPMAdapter
from archived.live_opm import LiveOPMHook, LiveOPMAdapter
from archived.common import optimizer


class OPMFixture(nn.Module):
    """One frozen query-row OPM, retaining mask, affine and denominator."""
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(16)
        self.linear_1, self.linear_2 = nn.Linear(16, 32), nn.Linear(16, 32)
        self.linear_out = nn.Linear(32 * 32, 128)
        self.eps = 1e-3

    def forward(self, m, mask):
        x = self.layer_norm(m)
        a, b = [proj(x)[0] * mask[0, :, None] for proj in [self.linear_1, self.linear_2]]
        outer = torch.einsum('ir,js->ijrs', a, b).flatten(-2)
        denominator = mask[0, :, None] * mask[0, None, :] + self.eps
        return self.linear_out(outer) / denominator[..., None]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260924)
    recipe = json.loads((HERE / 'recipe.json').read_text())
    lock = json.loads((ROOT / 'evidence/e1_execution_lock.json').read_text())
    manifest = json.loads((HERE.parent / 'source_manifest.json').read_text())
    def original_sha(path):
        # Anonymous packaging can redact host names in comments. Validate the
        # bundled bytes, then use its explicit original-snapshot hash binding.
        entry = manifest[str(path.relative_to(HERE.parent))]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        return entry.get('original_snapshot_sha256', entry['sha256'])
    for name in ['prospective_orientation.py', 'common.py']:
        assert original_sha(HERE / 'archived' / name) == lock['staging_hashes'][name]
    assert original_sha(HERE / 'archived/live_opm.py') == lock['source_hashes']['src/engramfold/models/live_opm.py']
    assert original_sha(HERE / 'rotations.npz') == lock['rotations_sha256']
    # The archived writer depends on these existing, matching source snapshots.
    for name in ['interface_heads.py', 'residue_predictors.py', 'native_geometry.py']:
        rel = 'src/engramfold/models/' + name
        assert original_sha(HERE.parent / rel) == lock['source_hashes'][rel]
    with np.load(HERE / 'rotations.npz') as arrays:
        transforms = {'I': torch.eye(128)}
        transforms.update({rid: torch.from_numpy(arrays[rid + '_fp32'].copy())
                           for rid in recipe['selected_rotations']})
    model = OPMFixture().eval().requires_grad_(False)
    frozen_before = copy.deepcopy(model.state_dict())
    length = 7
    features, m = torch.randn(length, 480), torch.randn(1, length, 16)
    mask = torch.ones(1, length)
    mask[0, -1] = 0
    baseline = model(m, mask).detach()
    target = baseline + 0.05 * torch.randn_like(baseline)
    results = {}
    for rid, rotation in transforms.items():
        torch.manual_seed(20260923)
        writer = ProspectiveOPMAdapter(rotation, learned_channels=True).eval()
        assert writer.channels.coordinates.numel() == 8001
        assert torch.equal(writer.channels.matrix(), torch.eye(128))
        opt = optimizer(writer, channels=True)
        assert [(g['lr'], g['weight_decay']) for g in opt.param_groups] == [(1e-4, .01), (1e-3, 0.)]
        assert all(g['betas'] == (.9, .999) and g['eps'] == 1e-8 for g in opt.param_groups)
        assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in writer.parameters()}
        assert sum(len(g['params']) for g in opt.param_groups) == len(list(writer.parameters()))
        hook = LiveOPMHook(model, writer, features)
        channel_gradients = []
        try:
            assert torch.equal(model(m, mask), baseline)
            for step in range(4):
                opt.zero_grad(set_to_none=True)
                prediction = model(m, mask)
                loss = (prediction - target).square().mean()
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in writer.parameters())
                channel_gradients.append(float(writer.channels.coordinates.grad.norm()))
                if step == 0:
                    assert channel_gradients[-1] == 0., 'Zero residual gives zero initial C gradient'
                torch.nn.utils.clip_grad_norm_(writer.parameters(), recipe['global_clip_norm'], error_if_nonfinite=True)
                opt.step()
        finally:
            hook.remove()
        assert any(g > 0 for g in channel_gradients[1:])
        with torch.no_grad():
            x = model.layer_norm(m)
            a, b = model.linear_1(x)[0], model.linear_2(x)[0]
            weight = model.linear_out.weight
            # Parent residual already applies R: the C wrapper must not apply it twice.
            rotated_delta = LiveOPMAdapter.residual(writer, features, a, b, weight, mask[0], model.eps)
            delta = writer.residual(features, a, b, weight, mask[0], model.eps)
            c = writer.channels.matrix()
            assert torch.allclose(delta, rotated_delta @ c.T, atol=1e-6, rtol=1e-5)
            assert torch.count_nonzero(delta) > 0
            assert torch.count_nonzero(delta[-1]) == torch.count_nonzero(delta[:, -1]) == 0
            orthogonality = float((c.T @ c - torch.eye(128)).norm())
            mean_error = float((c @ torch.ones(128) - 1).norm())
            assert orthogonality < 1e-3 and mean_error < 1e-3
            assert torch.allclose(delta.norm(dim=-1), rotated_delta.norm(dim=-1), atol=1e-6, rtol=1e-4)
            assert torch.allclose(delta.mean(-1), rotated_delta.mean(-1), atol=1e-6, rtol=1e-4)
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(v, frozen_before[k]) for k, v in model.state_dict().items())
        assert torch.equal(model(m, mask), baseline), 'Removing the hook restores the frozen path'
        results[rid] = dict(channel_gradient_norms=channel_gradients,
                           orthogonality_error=orthogonality, mean_direction_error=mean_error)
    environment = dict(python=platform.python_version(), pytorch=str(torch.__version__),
                       numpy=np.__version__, device=str(next(model.parameters()).device),
                       dtype=str(next(model.parameters()).dtype),
                       threads=torch.get_num_threads())
    print(json.dumps(dict(passed=True, conditions=results, exact_source_hash_checks=True,
                         environment=environment,
                         scope='CPU execution of sealed E2 writer, optimizer, zero initialization, injection, gradients and mask/constraint checks on synthetic inputs; not whole-protein prediction or quality reproduction.'), indent=2))


if __name__ == '__main__':
    main()
