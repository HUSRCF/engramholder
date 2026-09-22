"""Frozen learned-update diagnostics; labels never enter the writer."""
import json
from pathlib import Path
import torch
from engramfold.models.direction_extension import load_extension_writer
from engramfold.models.interface_heads import FrozenOPMDecoder
from engramfold.experiments.sequence_feature_cache import file_sha256, load_sequence_features


def signed_utility(gradient, delta):
    g,d=gradient.double(),delta.double()
    gn,dn=g.norm(),d.norm()
    if gn == 0 or dn == 0:
        return None
    return float(-(g*d).sum()/(gn*dn))


class FrozenStudents:
    def __init__(self, inventory, features, manifest, base_hash, experiment='E2'):
        self.inventory={k:v for k,v in json.loads(Path(inventory).read_text()).items() if v['experiment']==experiment}
        assert len(self.inventory)==12
        self.targets={v['target_id']:v for v in json.loads(Path(manifest).read_text())['targets']}
        self.features=Path(features);self.models={}
        feature_index=json.loads((self.features/'index.json').read_text())
        for name,item in self.inventory.items():
            assert file_sha256(Path(item['path']))==item['sha256']
            task=torch.load(item['path'],map_location='cpu',weights_only=False)
            assert task['protenix_checkpoint_sha256']==base_hash
            assert task['geometry']==item['geometry']
            for key in ['model_name','checkpoint_sha256','input_dim','layer']:
                assert task['feature_provenance'][key]==feature_index[key],'PLM provenance mismatch'
            writer=load_extension_writer(task,'cuda').requires_grad_(False).eval()
            c=task['common'];decoder=FrozenOPMDecoder(c['weight'],c['bias'],factor_dim=32,depth=c['depth'],eps=c['eps']).cuda()
            self.models[name]=(writer,decoder)
        self.target=None;self.updates={}

    @torch.no_grad()
    def prepare(self,target,query):
        if self.target==target:return self.updates
        features,provenance=load_sequence_features(self.features,[self.targets[target]])
        h=features[target].cuda();updates={}
        for name,(writer,decoder) in self.models.items():
            first=writer(h,decoder=decoder,**query)
            second=writer(h,decoder=decoder,**query)
            assert torch.equal(first,second),'nondeterministic frozen student'
            # Independent expanded expression checks the deployed fused formula.
            t,q=writer.components(h,decoder=decoder,query_a=query['query_a'],query_b=query['query_b'])
            expected=t+(q if writer.order=='full' else 0)
            residual=first-query['query_update']
            err=float((residual-expected).double().norm()/expected.double().norm().clamp_min(1e-30))
            assert err<1e-5,'student residual/rotation replay mismatch'
            updates[name]={'delta':residual.detach(),'replay_error':err,'rotation':writer.rotation_seed}
        self.target=target;self.updates=updates
        return updates

    @torch.no_grad()
    def evaluate(self,diag,uq,gradient,baseline,updates,oracle):
        gn=float(gradient.double().norm());un=float(uq.double().norm());eta=.001*un;rows=[]
        for name,item in updates.items():
            delta=item['delta'];dn=float(delta.double().norm());q=signed_utility(gradient,delta)
            row={'model':name,'rotation':item['rotation'],'residual_norm':dn,'norm_ratio':dn/un,'q_learned':q,'replay_error':item['replay_error']}
            if q is None:
                row.update(applicable=False);rows.append(row);continue
            direction=(delta.double()/dn).float()
            _,_,_,equal=diag.evaluate(uq+eta*direction)
            _,_,_,actual=diag.evaluate(uq+delta)
            row.update(applicable=True,equal=equal,actual=actual)
            qo=oracle.get(item['rotation'])
            row['q_oracle']=qo;row['relative_oracle_utility']=q/qo if qo is not None and abs(qo)>1e-12 else None
            for mode in ['fixed','dynamic']:
                row[mode+'_equal_decrease']=(baseline[mode+'_loss']-equal[mode+'_loss'])/(eta*gn)
                row[mode+'_actual_loss_decrease']=baseline[mode+'_loss']-actual[mode+'_loss']
            rows.append(row)
        return rows
