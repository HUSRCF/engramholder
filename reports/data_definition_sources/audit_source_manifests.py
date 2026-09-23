from pathlib import Path
import json,hashlib,collections,shutil
r=Path('/home/husrcf/Code/engramholder');exp=Path('/home/husrcf/Code/onestepfold/engramfold');src={
 'split96':r/'evidence/protenix_split96.json','split384':r/'evidence/protenix_split384.json','cache24':r/'evidence/protenix_cache_index.json',
 'dev8':exp/'artifacts/cross_backbone_20260921/data/dev8.json','train96':exp/'reports/folding_e2e_20260921/formal/train96.json','train384':exp/'reports/diamond_expansion_20260920/panel/train384.json',
 'confirmA':exp/'reports/independent_validation/manifest_v2.json','evaluation144':exp/'reports/folding_e2e_20260921/formal/evaluation144.json'}
d={k:json.loads(p.read_text()) for k,p in src.items()};a=d['split96'];b=d['split384'];s24=set(a['train24_ids']);s96=set(a['train96_ids']);s384=set(b['train384_ids']);assert s24<s96<s384
assert set(x['target_id'] for x in d['train96']['targets'])==s96
assert set(x['target_id'] for x in d['train384']['targets'])==s384
ca=d['confirmA']['targets'];cb=[x for x in d['evaluation144']['targets'] if x['panel']=='confirm96'];long=[x for x in d['evaluation144']['targets'] if x['panel']=='length48']
assert set(x['target_id'] for x in cb)==set(a['confirmation_ids'])
assert set(x['target_id'] for x in ca)==set(b['all_observed_confirmation_ids'])-set(a['confirmation_ids'])
assert hashlib.sha256(src['confirmA'].read_bytes()).hexdigest()==a['origins']['observed_manifest']
assert hashlib.sha256(src['train96'].read_bytes()).hexdigest()==a['files']['train96.json']
train=d['train384']['targets'];small=[x for x in train if x['target_id'] in s24];dev=d['dev8']['targets'];assert {x['target_id'] for x in dev}==set(a['dev8_ids']);assert hashlib.sha256(src['dev8'].read_bytes()).hexdigest()==a['files']['dev8.json']
groups={'Train24':small,'Train96':d['train96']['targets'],'Train384':train,'Dev8':dev,'Confirm96-A':ca,'Confirm96-B':cb,'Length48':long}
compact={}
for name,rows in groups.items():
 vals=[dict(target_id=x['target_id'],length=len(x['sequence']) if 'sequence'in x else x['length'],**({k:x[k] for k in ['length_bin','sequence_sha256','initial_release_date'] if k in x})) for x in rows]
 assert len(vals)==len({x['target_id'] for x in vals})
 compact[name]=dict(count=len(vals),actual_length=[min(x['length'] for x in vals),max(x['length'] for x in vals)],bins=dict(collections.Counter(x.get('length_bin','inherited') for x in vals)),records=vals)
for name in ['Dev8','Confirm96-A','Confirm96-B','Length48']:assert not s384&{x['target_id'] for x in groups[name]}
for rows in [ca,cb]:
 assert len({x['pdb_id'] for x in rows})==96
 for x in rows:assert x['initial_release_date']<='2021-09-30' and x['reference_ca_coverage']>=.9 and x['resolution_high_angstrom']<=2.5 and x['model_count']==1
for x in train:
 if x['target_id'] not in s24:assert x['initial_release_date']<='2021-09-30'
protocols={}
out=r/'reports/data_definition_sources';out.mkdir(exist_ok=True)
for name in ['independent_validation_v2','native_direction96_v1','train384_direction_v1']:
 p=exp/'docs'/f'{name}.md';shutil.copy2(p,out/p.name);protocols[name]=hashlib.sha256(p.read_bytes()).hexdigest()
assert protocols['independent_validation_v2']==d['confirmA']['protocol_sha256'];assert protocols['native_direction96_v1']==a['origins']['protocol']
for name,parent,sub in [('Train96-additions','Train96','Train24'),('Train384-additions','Train384','Train96')]:
 excluded={x['target_id'] for x in compact[sub]['records']}
 vals=[x for x in compact[parent]['records'] if x['target_id'] not in excluded]
 compact[name]=dict(count=len(vals),actual_length=[min(x['length'] for x in vals),max(x['length'] for x in vals)],bins=dict(collections.Counter(x['length_bin'] for x in vals)),records=vals)
result=dict(passed=True,scope='Manifest membership/length and stored metadata audit; no new alignment search, family annotation, CIF scoring or historical chronology certification.',datasets=compact,nested=True,train384_target_id_disjoint_from=['Dev8','Confirm96-A','Confirm96-B','Length48'],new_candidates_cutoff_not_inherited_train24_dev8=True,source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in src.items()},protocol_sha256=protocols)
(r/'evidence/data_composition_audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:{z:v[z] for z in ['count','actual_length','bins']} for k,v in compact.items()})
