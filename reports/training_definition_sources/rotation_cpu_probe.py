import torch, numpy as np, json, hashlib
from torch import Tensor
torch.set_num_threads(1)
def orthogonal_rotation(channels: int, seed: int | None) -> Tensor:
    if seed is None:
        return torch.eye(channels, dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(channels, channels, dtype=torch.float64, generator=gen)
    # The checked ROCm torch build lacks CPU LAPACK. NumPy provides the same
    # CPU float64 QR construction without changing or consuming the torch RNG.
    q, r = np.linalg.qr(matrix.numpy())
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return torch.from_numpy(q * signs[None, :]).float()
rows=[]
for seed in [20261001,20261002,20261003]:
 torch.manual_seed(123); before=torch.random.get_rng_state().clone(); nbefore=np.random.get_state()
 R=orthogonal_rotation(128,seed)
 after=torch.random.get_rng_state(); nafter=np.random.get_state()
 torch.manual_seed(987); R2=orthogonal_rotation(128,seed)
 rows.append(dict(seed=seed,sha256=hashlib.sha256(R.numpy().tobytes()).hexdigest(),dtype=str(R.dtype),device=str(R.device),determinant=float(np.linalg.det(R.double().numpy())),orthogonality_max_abs=float((R.double().T@R.double()-torch.eye(128,dtype=torch.float64)).abs().max()),torch_global_rng_unchanged=bool(torch.equal(before,after)),numpy_global_rng_unchanged=all(np.array_equal(x,y) for x,y in zip(nbefore,nafter)),independent_of_global_seed=bool(torch.equal(R,R2))))
print(json.dumps(dict(torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),rows=rows),indent=2))
