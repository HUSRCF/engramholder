"""CPU-only checks of the actual OPM adapter source, not a folding experiment."""
from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path(__file__).parent/'src'))
import torch
from engramfold.models.interface_heads import InterfaceHead,FrozenOPMDecoder
from engramfold.models.native_geometry import NativeGeometryHead
from engramfold.models.live_opm import LiveOPMAdapter

torch.set_num_threads(1);torch.manual_seed(24)
L=9;C=128;D=32
x=torch.randn(L,480);a,b=torch.randn(2,L,D);u=torch.randn(L,L,C)
decoder=FrozenOPMDecoder(torch.randn(C,D*D),torch.randn(C),factor_dim=D,depth=508.5,eps=.001)
query={'query_a':a,'query_b':b,'query_update':u,'decoder':decoder}
checks={}
for kind in ['factor','generic_plus']:
 h=InterfaceHead(kind);checks[kind+'_zero_baseline']=bool(torch.equal(h(x,**query),u))
for mode in ['full','tangent']:
 h=NativeGeometryHead(order=mode,rotation_seed=20261001)
 checks[mode+'_rotated_zero_baseline']=bool(torch.equal(h(x,**query),u))
# Use the real live OpenFold G+ branch with a matched reparameterization.
h=LiveOPMAdapter(backbone='openfold',kind='generic_plus').double()
with torch.no_grad():
 h.output_head[-1].weight.normal_(std=.01);h.output_head[-1].bias.normal_(std=.01)
r=copy.deepcopy(h);q=torch.linalg.qr(torch.randn(C,C,dtype=torch.float64)).Q
with torch.no_grad():
 r.rotate_generic_output=True
 r.rotation.copy_(q);r.output_head[-1].weight.copy_(q.T@h.output_head[-1].weight);r.output_head[-1].bias.copy_(q.T@h.output_head[-1].bias)
w=decoder.weight.double();mask=torch.ones(L,dtype=torch.float64)
y=h.residual(x.double(),a.double(),b.double(),w,mask)
z=r.residual(x.double(),a.double(),b.double(),w,mask)
error=float((y-z).abs().max().detach());checks['gplus_absorbable_output_rotation']=error<1e-10
assert all(checks.values()),checks
print(json.dumps({'passed':True,'checks':checks,'gplus_max_absolute_error':error,'scope':'operator-level numerical check; no structure inference or adaptation-result replication'},indent=2))
