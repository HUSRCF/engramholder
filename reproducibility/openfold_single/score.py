"""Score a full prediction CIF with the locked reference correspondence.

Independent process: only this scoring command reads the experimental reference.
Pair-lDDT matches the paper's fixed reference pairs, thresholds and missing rule.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np


def ca_coordinates(path, chain):
    text=gzip.open(path,'rt').read() if path.suffix=='.gz' else path.read_text()
    block=gemmi.cif.read_string(text).sole_block()
    columns=['group_PDB','label_asym_id','label_seq_id','label_atom_id','Cartn_x','Cartn_y','Cartn_z','pdbx_PDB_model_num','label_alt_id']
    found={};priority={}
    for row in block.find(['_atom_site.'+k for k in columns]):
        if row[0]!='ATOM' or row[1]!=chain or row[2] in ['.','?'] or row[3]!='CA' or row[7] not in ['1','.','?']: continue
        i=int(row[2]);rank=0 if row[8] in ['.','?','A'] else 1
        if i in found and rank>=priority[i]: continue
        xyz=np.array([float(row[k]) for k in (4,5,6)])
        if np.isfinite(xyz).all(): found[i]=xyz;priority[i]=rank
    assert found
    return found


def score(ref,pred):
    ids=sorted(ref);r=np.array([ref[i] for i in ids]);valid=np.array([i in pred for i in ids])
    x=np.array([pred.get(i,np.zeros(3)) for i in ids]);dr=np.linalg.norm(r[:,None]-r[None],axis=-1)
    dx=np.linalg.norm(x[:,None]-x[None],axis=-1);pairs=(dr>0)&(dr<15)
    available=valid[:,None]&valid[None];error=np.abs(dr-dx)
    credit=sum(((error<t)&available).astype(float) for t in [.5,1.,2.,4.])/4
    mask=np.triu(pairs,1);assert mask.any()
    return dict(ca_pair_lddt=float(credit[mask].mean()),reference_ca_count=len(ids),
                prediction_reference_coverage=float(valid.mean()),reference_pair_count=int(mask.sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--mapping',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    m=json.loads(a.mapping.read_text());assert hashlib.sha256(a.reference.read_bytes()).hexdigest()==m['raw_mmcif_sha256']
    predicted_block=gemmi.cif.read_file(str(a.prediction)).sole_block()
    predicted_sequence=gemmi.cif.as_string(predicted_block.find_value('_entity_poly.pdbx_seq_one_letter_code_can')).replace('\n','').replace(' ','')
    assert predicted_sequence==m['sequence'], 'Prediction sequence differs from reference mapping'
    ref=ca_coordinates(a.reference,m['source_label_asym_id']);assert sorted(ref)==m['reference_ca_indices']
    pred=ca_coordinates(a.prediction,'A');assert sorted(pred)==list(range(1,len(m['sequence'])+1))
    result=score(ref,pred);result.update(prediction_sha256=hashlib.sha256(a.prediction.read_bytes()).hexdigest(),
        reference_sha256=m['raw_mmcif_sha256'],metric_scope='Primary fixed-reference C-alpha pair-lDDT; no TM-score claim')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
