"""Native AtlasFold training tensors, with labels kept outside sequence features."""
from pathlib import Path
import gzip,hashlib
import numpy as np
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def state_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        h.update(name.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_model(root):
    from atlasfold.pretrained import load_model as load
    from atlasfold.train.monomer.model_train import AtlasFoldForTrain
    model=load('atlasfold',device='cuda',kernel='torch',model_path=str(root/'weights/atlasfold/snapshot/weights/atlasfold-260703.pth'),lm_path=str(root/'weights/atlaslm3b/snapshot/weights/atlaslm_3b_base.pth')).requires_grad_(False)
    model.__class__=AtlasFoldForTrain
    for module in model.modules():
        if hasattr(module,'blocks_per_ckpt'):module.blocks_per_ckpt=1
    return model


def training_mode(model):
    model.train();model.lm.eval()
    for module in model.modules():
        if 'Dropout' in module.__class__.__name__:module.eval()


def prepare(root,t):
    from atlasfold.common import featurize,residue_constants as rc
    seq=t['sequence'];L=len(seq)
    feat={k:torch.from_numpy(v)[None].cuda() for k,v in featurize.featurize(seq,pad_to_multiple_of=4).items()}
    cache=torch.load(root/'data'/f"{t['target_id']}.pt",map_location='cpu',weights_only=False)
    assert cache['sequence_sha256']==hashlib.sha256(seq.encode()).hexdigest()
    esm=cache['features'].float().cuda();assert esm.shape==(L,480)
    path=root/'data'/f"{t['target_id']}.cif.gz"
    with gzip.open(path,'rt') as f:cif=MMCIF2Dict(f)
    xyz,mask=atom14_labels(cif,t,rc)
    label={'coordinates':torch.from_numpy(xyz)[None].cuda(),'resolved_mask':torch.from_numpy(mask)[None].cuda()}
    return feat,esm,label


def native_forward_loss(model,feat,label,seed):
    from atlasfold.model import SamplingConfig
    from atlasfold.train.losses.distogram import DistogramLoss
    from atlasfold.train.losses.diffusion import MSELoss,SmoothLDDTLoss
    torch.manual_seed(seed)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model.forward_train({k:v.clone() for k,v in feat.items()},label,num_recycles=3,diffusion_batch_size=1,train_trunk=True,train_diffusion_head=True,train_confidence_head=False,sampling_config=SamplingConfig())
    d=pred['distogram'];q=pred['diffusion']
    dl=DistogramLoss()(d['logits'],d['boundaries'],label['coordinates'],label['resolved_mask'],feat['pseudo_beta']).mean()
    ml=(MSELoss()(q['x_out'],label['coordinates'],label['resolved_mask'])*q['loss_weights']).mean()
    sl=SmoothLDDTLoss(chunk_size=1,use_kernel=False)(q['x_out'],label['coordinates'],label['resolved_mask']).mean()
    return .4*dl+2*(ml+sl),{'distogram':dl,'mse':ml,'smooth_lddt':sl},q['x_out']


def atom14_labels(cif,t,rc):
    """Use full entity numbering; unresolved residues remain masked, never cropped."""
    seq=t['sequence'];L=len(seq)
    xyz=np.zeros((L,14,3),np.float32);mask=np.zeros((L,14),bool);n=len(cif['_atom_site.label_atom_id'])
    for k,name in enumerate(cif['_atom_site.label_atom_id']):
        if cif['_atom_site.label_asym_id'][k]!=t['source_label_asym_id']:continue
        if cif.get('_atom_site.pdbx_PDB_model_num',['1']*n)[k]!='1':continue
        try:i=int(cif['_atom_site.label_seq_id'][k])-1
        except ValueError:continue
        if not 0<=i<L:continue
        res=rc.restype_1to3[seq[i]]
        observed=cif['_atom_site.label_comp_id'][k]
        if observed!=res:
            # Modified residues lack this canonical atom14 label; canonical
            # disagreement is a mapping error, not a reason to shift indices.
            if observed in rc.restype_1to3.values():
                raise ValueError(f"Canonical sequence mismatch at {i+1}: {observed} != {res}")
            continue
        j=rc.restype_atom14_order[res].get(name)
        if j is None or mask[i,j]:continue
        coord=[float(cif['_atom_site.Cartn_'+ax][k]) for ax in 'xyz'];assert np.isfinite(coord).all()
        xyz[i,j]=coord;mask[i,j]=True
    assert mask[:,1].sum()>=3, "Insufficient resolved CA atoms"
    xyz-=xyz[mask].mean(0);xyz[~mask]=0
    # Native atom-attention windows require multiples of four. Preserve all
    # real residues; padded labels are unresolved and ESM features remain L.
    pad=(-L)%4
    xyz=np.pad(xyz,((0,pad),(0,0),(0,0)))
    mask=np.pad(mask,((0,pad),(0,0)),constant_values=False)
    return xyz,mask
