"""Live native product/difference residuals; AtlasFold keeps its native AtlasLM."""
import torch
from torch import nn
from .interface_heads import InterfaceHead
from .native_geometry import orthogonal_rotation


def prod_diff_residual(a,b,da,db,weight,rotation=None):
    if a.shape!=b.shape or a.shape!=da.shape or a.shape!=db.shape:
        raise ValueError('Factor shapes must match')
    if weight.shape[-1]!=2*a.shape[-1]:raise ValueError('Decoder shape mismatch')
    difference=da[:,None,:]-db[None,:,:]
    product=da[:,None,:]*b[None,:,:]+a[:,None,:]*db[None,:,:]+da[:,None,:]*db[None,:,:]
    w=weight if rotation is None else rotation.to(weight)@weight
    return nn.functional.linear(torch.cat((difference,product),-1),w)


class LiveProdDiffAdapter(InterfaceHead):
    def __init__(self,kind='factor',rotation_seed=None,**kwargs):
        if kind not in ['factor','generic_plus']:raise ValueError('Unsupported arm')
        if kind!='factor' and rotation_seed is not None:raise ValueError('Rotation only for factor')
        super().__init__(kind,factor_dim=128,pair_channels=128,**kwargs)
        self.register_buffer('rotation',orthogonal_rotation(128,rotation_seed))

    def residual(self,features,a,b,weight):
        hidden=self.hidden(features)
        if self.kind=='factor':
            da,db=self.output_head(hidden).chunk(2,-1)
            return prod_diff_residual(a,b,da,db,weight,self.rotation)
        nodes=self.node(hidden);columns=torch.arange(len(a),device=a.device);chunks=[]
        for start in range(0,len(a),32):
            rows=torch.arange(start,min(start+32,len(a)),device=a.device)
            i,j=torch.meshgrid(rows,columns,indexing='ij');chunks.append(self._pairs(nodes,i,j,a,b))
        return torch.cat(chunks,0)


class LiveProdDiffHook:
    def __init__(self,module,writer,features):
        self.writer=writer;self.features=features;self.calls=[]
        self.handle=module.register_forward_hook(self._hook)
    def _hook(self,module,args,baseline):
        s=args[0]
        if s.ndim!=3 or s.shape[0]!=1:raise ValueError('One full chain per invocation')
        n=len(self.features)
        if n>s.shape[1]:raise ValueError('Truncated sequence')
        # Padding is inference bucketing only. Encoder processes real residues only.
        aa,bb=module.linear_in(module.layernorm(s[:,:n])).chunk(2,-1)
        if aa.shape[-1]!=128 or baseline.shape[-1]!=128:raise ValueError('Unsealed interface dimensions')
        delta=self.writer.residual(self.features,aa[0],bb[0],module.linear_out.weight)
        pad=s.shape[1]-n
        delta=nn.functional.pad(delta,(0,0,0,pad,0,pad))[None]
        self.calls.append(dict(length=n,padded_length=s.shape[1],grad_enabled=torch.is_grad_enabled()))
        return baseline+delta
    def remove(self):self.handle.remove()
