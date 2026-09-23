"""One FASTA to full-atom CIF. No reference structures or target labels are read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'src'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_fasta(path):
    lines=Path(path).read_text().splitlines()
    if sum(x.startswith('>') for x in lines)!=1 or not lines[0].startswith('>'):
        raise ValueError('Exactly one FASTA record required')
    seq=''.join(x.strip() for x in lines[1:]).upper()
    if not seq or not set(seq)<=set('ACDEFGHIKLMNPQRSTVWY'):
        raise ValueError('Only the 20 standard amino acids are supported')
    return seq


def write_cif(path, sequence, xyz, mask):
    import gemmi
    from openfold.np import residue_constants as rc
    doc=gemmi.cif.Document();b=doc.add_new_block('prediction')
    b.set_pair('_entry.id','prediction')
    entity=b.init_loop('_entity_poly.', ['entity_id','type','pdbx_seq_one_letter_code_can'])
    entity.add_row(['1',gemmi.cif.quote('polypeptide(L)'),gemmi.cif.quote(sequence)])
    loop=b.init_loop('_atom_site.', ['group_PDB','id','type_symbol','label_atom_id','label_alt_id',
        'label_comp_id','label_asym_id','label_entity_id','label_seq_id','pdbx_PDB_ins_code',
        'Cartn_x','Cartn_y','Cartn_z','occupancy','B_iso_or_equiv','auth_seq_id','auth_comp_id',
        'auth_asym_id','auth_atom_id','pdbx_PDB_model_num'])
    atom_id=0
    for i,aa in enumerate(sequence):
        name=rc.restype_1to3[aa]
        for j,atom in enumerate(rc.atom_types):
            if not mask[i,j]: continue
            atom_id+=1
            loop.add_row(['ATOM',str(atom_id),atom[0],atom,'.',name,'A','1',str(i+1),'?',
                          *[f'{v:.9f}' for v in xyz[i,j]],'1.00','0.00',str(i+1),name,'A',atom,'1'])
    doc.write_file(str(path))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fasta',required=True,type=Path)
    p.add_argument('--af2-weights',required=True,type=Path)
    p.add_argument('--esm-weights',required=True,type=Path)
    p.add_argument('--adapter',type=Path,default=ROOT/'adapter.pt')
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    import numpy as np
    import torch
    import esm
    assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert Path(esm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    from engramfold.experiments.openfold_adapter_runtime import runtime_config,sequence_features
    from engramfold.models.live_opm import LiveOPMAdapter,LiveOPMHook
    from openfold.model.model import AlphaFold
    from openfold.utils.import_weights import assign,generate_translation_dict,process_translation_dict
    cfglock=json.loads((ROOT/'asset_manifest.json').read_text())
    assert sha(a.af2_weights)==cfglock['af2_sha256']
    assert sha(a.esm_weights)==cfglock['esm_sha256']
    assert sha(a.adapter)==cfglock['adapter_sha256']
    if a.out.exists(): raise FileExistsError('Use a new output directory; do not overwrite a run')
    a.out.mkdir(parents=True)
    seq=read_fasta(a.fasta);tick=time.monotonic()
    assert torch.cuda.is_available()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(cfglock['seed'])
    with torch.serialization.safe_globals([argparse.Namespace]):
        payload=torch.load(a.esm_weights,map_location='cpu',weights_only=True)
    plm,alphabet=esm.pretrained.load_model_and_alphabet_core(cfglock['esm_name'],payload)
    plm=plm.cuda().eval().requires_grad_(False);del payload
    _,_,tokens=alphabet.get_batch_converter()([('input',seq)])
    assert tokens.shape==(1,len(seq)+2)
    with torch.no_grad():
        features=plm(tokens.cuda(),repr_layers=[12],return_contacts=False)['representations'][12][0,1:-1].float()
    assert features.shape==(len(seq),480) and torch.isfinite(features).all()
    torch.save(features.cpu(),a.out/'features.pt')
    del plm,tokens;torch.cuda.empty_cache()
    cfg=runtime_config(activation_checkpointing=False)
    torch.manual_seed(cfglock['seed']);inputs=sequence_features(seq,cfg)
    model=AlphaFold(cfg).eval().requires_grad_(False)
    weights=np.load(a.af2_weights)
    translation=process_translation_dict(generate_translation_dict(model,'model_3_ptm',is_multimer=False))
    assert set(translation)==set(weights.files);assign(translation,weights)
    del weights,translation
    model.cuda();before=state_hash(model)
    assert before==cfglock['frozen_backbone_sha256']
    ck=torch.load(a.adapter,map_location='cpu',weights_only=True)
    writer=LiveOPMAdapter(backbone='openfold',kind=ck['kind'],rotation_seed=ck['rotation_seed'])
    writer.load_state_dict(ck['writer']);writer=writer.cuda().eval().requires_grad_(False)
    hook=LiveOPMHook(model.evoformer.blocks[0].outer_product_mean,writer,features)
    calls=[];counter=model.evoformer.register_forward_hook(lambda *args:calls.append(1))
    torch.manual_seed(cfglock['seed'])
    try:
        with torch.no_grad(): result=model({k:v.cuda() for k,v in inputs.items()})
    finally: hook.remove();counter.remove()
    assert len(calls)==len(hook.calls)==4
    xyz=result['final_atom_positions'].cpu().numpy();mask=result['final_atom_mask'].cpu().numpy()
    assert xyz.shape==(len(seq),37,3) and np.isfinite(xyz).all() and mask[:,1].sum()==len(seq)
    assert state_hash(model)==before and all(p.grad is None for p in model.parameters())
    np.savez(a.out/'prediction.npz',coordinates=xyz,mask=mask,sequence=seq)
    write_cif(a.out/'prediction.cif',seq,xyz,mask)
    import openfold
    report=dict(complete=True,sequence_length=len(seq),sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),
        input_fasta_sha256=sha(a.fasta),af2_sha256=sha(a.af2_weights),esm_sha256=sha(a.esm_weights),
        adapter_sha256=sha(a.adapter),trunk_calls=len(calls),opm_calls=hook.calls,
        frozen_backbone_sha256=before,features_recomputed=True,reference_read=False,
        cif_sha256=sha(a.out/'prediction.cif'),seconds=time.monotonic()-tick,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),python=sys.version,
        interpreter=sys.executable,environment_prefix=sys.prefix,base_prefix=sys.base_prefix,
        openfold_path=openfold.__file__,torch_path=torch.__file__,esm_path=esm.__file__,
        pythonpath=os.environ.get('PYTHONPATH'))
    (a.out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
