"""Locked mean-preserving conjugate rotations and shared orthogonal compensation."""
from __future__ import annotations
import itertools
import numpy as np
import torch
from torch import nn
from .live_opm import LiveOPMAdapter


def mean_basis(n=128):
    u=np.ones(n,dtype=np.float64)/np.sqrt(n)
    v=np.eye(n,dtype=np.float64)[:,0]-u;v/=np.linalg.norm(v)
    return u,(np.eye(n)-2*np.outer(v,v))[:,1:]


def special_orthogonal(n,seed):
    q,r=np.linalg.qr(np.random.default_rng(seed).standard_normal((n,n)))
    signs=np.sign(np.diag(r));signs[signs==0]=1;q=q*signs[None,:]
    if np.linalg.det(q)<0:q[:,-1]*=-1
    return q


def conjugate_family(n=128,base_seed=20270100,seeds=range(20270101,20270109)):
    u,B=mean_basis(n);T=special_orthogonal(n-1,base_seed)
    matrices={int(seed):np.outer(u,u)+B@special_orthogonal(n-1,seed)@T@special_orthogonal(n-1,seed).T@B.T for seed in seeds}
    return u,B,T,matrices


class SharedOrthogonalChannels(nn.Module):
    def __init__(self,n=128):
        super().__init__();_,B=mean_basis(n)
        self.register_buffer('basis',torch.from_numpy(B).float())
        self.register_buffer('upper',torch.triu_indices(n-1,n-1,offset=1))
        self.coordinates=nn.Parameter(torch.zeros((n-1)*(n-2)//2))
    def matrix(self):
        d=self.basis.shape[1];a=self.coordinates.new_zeros((d,d))
        a[self.upper[0],self.upper[1]]=self.coordinates
        skew=a-a.T
        # Exact identity at zero, algebraically uu^T+B exp(A) B^T.
        return torch.eye(d+1,device=a.device,dtype=a.dtype)+self.basis@(torch.matrix_exp(skew)-torch.eye(d,device=a.device,dtype=a.dtype))@self.basis.T
    def forward(self,x):return x@self.matrix().to(x).T


class ProspectiveOPMAdapter(LiveOPMAdapter):
    def __init__(self,rotation,learned_channels=False):
        super().__init__(backbone='openfold',kind='factor',rotation_seed=None)
        if rotation.shape!=(128,128):raise ValueError('Expected locked 128-channel rotation')
        self.rotation.copy_(rotation.to(self.rotation))
        self.channels=SharedOrthogonalChannels()if learned_channels else None
    def residual(self,*args,**kwargs):
        delta=super().residual(*args,**kwargs)
        return self.channels(delta)if self.channels is not None else delta


def average_ranks(values):
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite diagnostic')
    order=np.argsort(x,kind='stable');r=np.empty(len(x),float);start=0
    while start<len(x):
        end=start+1
        while end<len(x)and x[order[end]]==x[order[start]]:end+=1
        r[order[start:end]]=(start+1+end)/2;start=end
    return r


def spearman(x,y):
    a,b=average_ranks(x),average_ranks(y);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den)if den else None


def exact_rotation_test(x,y):
    if len(x)!=8 or len(y)!=8:raise ValueError('Exactly eight rotation units required')
    rho=spearman(x,y)
    if rho is None:return dict(status='constant_diagnostic_or_outcome',rho=None,p_one_sided=None,n_rotations=8)
    a,b=average_ranks(x),average_ranks(y);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    perm=np.asarray(list(itertools.permutations(range(8))),dtype=np.int64)
    null=b[perm]@a/den
    return dict(status='ok',rho=rho,p_one_sided=float(np.mean(null>=rho-1e-12)),n_rotations=8,permutations=40320)
