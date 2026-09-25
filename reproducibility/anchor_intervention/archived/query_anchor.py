"""E3: replace only the residual anchor, retaining the live OPM baseline.

Four reference anchors are separate from the adapted forward.  The final
anchor is explicitly pinned during nonreentrant checkpoint recomputation.
"""
from __future__ import annotations

import hashlib
import torch


def tensor_hash(items):
    h = hashlib.sha256()
    for name, value in sorted(items.items()):
        v = value.detach().cpu().contiguous()
        h.update(str((name, str(v.dtype), tuple(v.shape))).encode())
        h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def opm_factors(module, args, kwargs):
    m = args[0] if args else kwargs['m']
    mask = args[1] if len(args) > 1 else kwargs.get('mask')
    if mask is None:
        mask = m.new_ones(m.shape[:-1])
    if m.ndim != 3 or m.shape[0] != 1:
        raise ValueError('one complete chain and one query MSA row required')
    ln = module.layer_norm(m)
    n = m.shape[-2]
    return dict(a=module.linear_1(ln).reshape(n, -1),
                b=module.linear_2(ln).reshape(n, -1),
                mask=mask.reshape(n), eps=float(module.eps))


class QueryAnchorCapture:
    def __init__(self, module):
        self.anchors = []
        self.calls = []
        self.handle = module.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, baseline):
        if len(self.anchors) >= 4:
            raise RuntimeError('reference must have exactly four forward calls')
        anchor = opm_factors(module, args, kwargs)
        self.anchors.append({k: v.detach().clone() if torch.is_tensor(v) else v
                             for k, v in anchor.items()})
        self.calls.append(torch.is_grad_enabled())
        # A capture never replaces the actual output.

    def remove(self):
        self.handle.remove()


class QueryAnchorHook:
    def __init__(self, module, writer, features, anchors, *, diagnostics=False):
        if len(anchors) != 4:
            raise ValueError('four recycle-specific reference anchors required')
        if writer.backbone != 'openfold' or writer.channels is not None:
            raise ValueError('E3 uses ordinary OpenFold Factor, without E2 C')
        self.writer, self.features, self.anchors = writer, features, anchors
        self.calls = []
        self.backward_phase = False
        self.diagnostics = diagnostics
        self.handle = module.register_forward_hook(self._hook, with_kwargs=True)

    def finish_forward(self):
        if len(self.calls) != 4:
            raise RuntimeError('missing or extra forward recycle')
        self.backward_phase = True

    def _hook(self, module, args, kwargs, baseline):
        index = 3 if self.backward_phase else len(self.calls)
        if index >= 4 or (self.backward_phase and not torch.is_grad_enabled()):
            raise RuntimeError('invalid checkpoint replay')
        q = self.anchors[index]
        m = args[0] if args else kwargs['m']
        mask = args[1] if len(args) > 1 else kwargs.get('mask')
        mask = m.new_ones(m.shape[:-1]) if mask is None else mask
        if m.ndim != 3 or m.shape[0] != 1 or m.shape[-2] != len(self.features):
            raise ValueError('input shape changed')
        if not torch.equal(mask.reshape(-1), q['mask']) or float(module.eps) != q['eps']:
            raise ValueError('reference mask/normalization mismatch')
        if q['a'].requires_grad or q['b'].requires_grad:
            raise ValueError('reference must be detached')
        residual = self.writer.residual(self.features, q['a'], q['b'],
                                        module.linear_out.weight, q['mask'], q['eps'])
        if baseline.shape != residual.shape:
            raise ValueError('baseline shape changed')
        row = dict(anchor_index=index, grad_enabled=torch.is_grad_enabled(),
                   backward_replay=self.backward_phase)
        if self.diagnostics and not self.backward_phase:
            row['residual_norm'] = float(residual.detach().double().norm())
            row['baseline_norm'] = float(baseline.detach().double().norm())
        self.calls.append(row)
        return baseline + residual

    def remove(self):
        self.handle.remove()


def query_reference(model, features, *, gradient_path, seed):
    """No labels/writer. Preserve outer RNG; require RNG-free reference forward."""
    if any(m.training for m in model.modules()):
        raise RuntimeError('the locked eval/dropout-off path is required')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('reference backbone must be frozen')
    module = model.evoformer.blocks[0].outer_product_mean
    if module._forward_hooks:
        raise RuntimeError('reference cannot run with an adapter hook installed')
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    capture = QueryAnchorCapture(module)
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            cpu = torch.get_rng_state().clone()
            gpu = torch.cuda.get_rng_state(device).clone() if devices else None
            with torch.set_grad_enabled(gradient_path):
                output = model(features)
            if not torch.equal(cpu, torch.get_rng_state()):
                raise RuntimeError('reference consumed CPU RNG; target-only cache is invalid')
            if devices and not torch.equal(gpu, torch.cuda.get_rng_state(device)):
                raise RuntimeError('reference consumed CUDA RNG; target-only cache is invalid')
            coordinates = output['final_atom_positions'].detach().clone()
            del output
    finally:
        capture.remove()
    expected = [False, False, False, bool(gradient_path)]
    if capture.calls != expected:
        raise RuntimeError(f'incorrect recycle gradient path: {capture.calls}')
    for q in capture.anchors:
        if any(not torch.isfinite(q[k]).all() for k in ['a', 'b', 'mask']):
            raise RuntimeError('nonfinite query anchor')
    return capture.anchors, coordinates
