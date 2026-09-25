import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from common import read, write, save, sha, contract


def check(root, gpu=False):
    lock = read(root / 'repetition_lock.json')
    assert sha(root / 'execution_lock.json') == lock['parent_execution_lock_sha256']
    assert sha(root / 'repetition_protocol.md') == lock['protocol_sha256']
    for name, digest in lock['files'].items():
        assert sha(root / 'repetition' / name) == digest, name
    assert root.resolve() != Path(lock['parent_root']).resolve()
    old, old_sha = contract(root, gpu)
    return lock, old, old_sha


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def rng():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def flat_error(a, b):
    aa = torch.cat([v.detach().cpu().reshape(-1).double() for v in a])
    bb = torch.cat([v.detach().cpu().reshape(-1).double() for v in b])
    return dict(relative=float((aa-bb).norm()/aa.norm().clamp_min(1e-30)),
                maxabs=float((aa-bb).abs().max()))
