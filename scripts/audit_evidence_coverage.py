"""Read-only score audit; no inference, re-scoring, new CI or model selection."""
import json, hashlib, shutil, re, statistics
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=Path('/home/husrcf/Code/onestepfold/engramfold')
E=ROOT/'evidence'
copies={
'protenix_direction_lock.json':'reports/native_direction96_20260919/execution_lock.json',
'protenix_direction_completion.json':'reports/native_direction96_20260919/completion_audit.json',
'protenix_train384_training_lock.json':'reports/diamond_expansion_20260920/training_lock.json',
'protenix_train384_eval_lock.json':'reports/diamond_expansion_20260920/evaluation_lock.json',
'protenix_gplus.json':'reports/access_panel_followup_20260920/metric_analysis/analysis.json',
'protenix_gplus_records.json':'reports/access_panel_followup_20260920/metric_analysis/metric_records.json',
'protenix_gplus_lock.json':'reports/access_panel_followup_20260920/evaluation_lock.json',
'protenix_cache_index.json':'reports/plm_interface_control/cache_index.json',
'protenix_split96.json':'reports/native_direction96_20260919/split_lock.json',
'protenix_split384.json':'reports/diamond_expansion_20260920/panel/split_lock.json',
'cumulative_exposure.json':'reports/diamond_expansion_20260920/cumulative_exposure_scope.json',
'length48_records.json':'reports/length48_v2_20260920/metric_analysis/metric_records.json',
'length48_selection_lock.json':'reports/length48_v2_20260920/panel/selection_lock.json',
'length48_manifest.json':'reports/length48_v2_20260920/manifest.json',
'length48_evaluation_lock.json':'reports/length48_v2_20260920/evaluation_lock.json',
'length48_supplement_gate.json':'reports/length48_v2_20260920/supplement_gate.json',
'openfold_train96_records.json':'reports/openfold_followup_20260921/train96/metric_records.json',
'openfold_train384_records.json':'reports/openfold_followup_20260921/train384/metric_records.json',
}
for dst,src in copies.items(): shutil.copy2(SRC/src,E/dst)
def load(n):return json.loads((E/(n+'.json')).read_text())
def save(n,d): (ROOT/'reports'/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def pro(n):return {k:v['records'] for k,v in load(n)['systems'].items()}
P=pro('protenix_direction');X=pro('protenix_extensions');Z=pro('protenix_train384');L=load('length48_records');G=load('protenix_gplus_records')
contrasts=[]
def add(label,systems,natives,controls,official,metric='ca_lddt',target_ids=None):
 assert len(natives)==3 and len(controls) in (3,9)
 ids=target_ids or [r['target_id'] for r in systems[natives[0]]]
 assert len(set(ids))==len(ids)
 arr={}
 for k in natives+controls:
  rows=systems[k]; by={r['target_id']:r for r in rows}
  assert len(rows)==len(by)==len(ids) and set(by)==set(ids),(label,k)
  # Original failed scores remain in original arrays; no success-only filtering.
  arr[k]=np.array([by[i][metric] for i in ids],float)
  assert np.all(np.isfinite(arr[k]))
 n=np.array([arr[k] for k in natives]);r=np.array([arr[k] for k in controls]);d=n.mean(0)-r.mean(0)
 expected=np.array(official['per_target'])
 err=float(np.max(abs(d-expected)))
 assert err<1e-12,(label,err)
 assert abs(d.mean()-official['mean'])<1e-12
 if len(controls)==9:
  # controls ordered rotation-major, seed-minor.
  rr=r.reshape(3,3,len(ids)); seed=(n-rr.mean(0)).mean(1);rotation=(n.mean(0)-rr.mean(1)).mean(1)
  grid=(n[None,:,:]-rr).mean(2)
 else:seed=(n-r).mean(1);rotation=None;grid=None
 out=dict(label=label,n=len(ids),native_mean=float(n.mean()),control_mean=float(r.mean()),mean=float(d.mean()),ci95=official.get('ci95'),positive=int((d>0).sum()),zero=int((d==0).sum()),quantiles={str(q):float(np.quantile(d,q)) for q in [0,.1,.25,.5,.75,.9,1]},per_seed=seed.tolist(),per_rotation=None if rotation is None else rotation.tolist(),seed_rotation_grid=None if grid is None else grid.tolist(),per_target=dict(zip(ids,d.tolist())),native_systems=natives,control_systems=controls,max_per_target_difference_from_original=err)
 if grid is not None:out['positive_seed_rotation_cells']=int((grid>0).sum())
 contrasts.append(out)
s=[20260923,20260924,20260925];rot=[20261001,20261002,20261003]
def names(size,order,steps):return [f'n{size}_{order}_native_s{v}_u{steps}' for v in s], [f'n{size}_{order}_r{r}_s{v}_u{steps}' for r in rot for v in s]
for order in ['tangent','full']:
 n,r=names(24,order,384);o=load('protenix_direction')['primary'] if order=='tangent' else load('protenix_direction')['secondary']['native_full_minus_rotated_full_n24_u384'];add('Mini Train24 '+order+' / Confirm96-B',P,n,r,o)
n,r=names(96,'full',1536);r=[f'C_mini_ordinary_r{a}_s{b}' for a in rot for b in s]
add('Mini Train96 full / Confirm96-B',X,n,r,load('protenix_extensions')['secondary']['mini_train96_full_native_minus_rotated'])
n,r=names(384,'full',1536);add('Mini Train384 full / Confirm96-B',Z,n,r,load('protenix_train384')['primary'])
add('Mini Train384 full / Length48',L,n,r,load('length48')['metrics']['ca_pair_lddt']['primary'],'ca_pair_lddt',load('length48')['targets'])
add('Tiny Train24 tangent / Confirm96-B',X,[f'A_tiny_ordinary_rNone_s{v}' for v in s],[f'A_tiny_ordinary_r{a}_s{b}' for a in rot for b in s],load('protenix_extensions')['primary'])
add('Mini Train24 tangent mean-preserving / Confirm96-B',X,[f'n24_tangent_native_s{v}_u384' for v in s],[f'B_mini_mean_r{a}_s{b}' for a in [20261011,20261012,20261013] for b in s],load('protenix_extensions')['secondary']['mini_native_minus_mean_preserving'])
add('Mini Train24 full vs G+ / Confirm96-B (separate recipe)',G,[f'factor_s{v}_u384' for v in [20260920,20260921,20260922]],[f'generic_plus_s{v}_u384' for v in [20260920,20260921,20260922]],load('protenix_gplus')['metrics']['ca_pair_lddt'],'ca_pair_lddt',load('protenix_gplus')['targets'])
for size in [96,384]:
 raw=load(f'openfold_train{size}_records');summary=load(f'openfold_train{size}')
 for panel in ['confirm96','length48']:
  systems={k:[v for v in raw if v['panel']==panel and v['system']==k] for k in {v['system'] for v in raw}}
  o=summary['panels'][panel]['ca_lddt']['native_minus_rotated']
  add(f'OpenFold Train{size} full / {panel}',systems,[f'native_s{v}' for v in s],[f'r{a}_s{b}' for a in rot for b in s],o,target_ids=o['target_ids'])
# Verify matched-construction interaction directly from target arrays, not intervals.
d384=contrasts[3]['per_target'];d96=contrasts[2]['per_target'];interaction={i:d384[i]-d96[i] for i in d384}
orig=load('protenix_data_interaction')['direction_difference_in_differences']
assert np.max(abs(np.array(list(interaction.values()))-np.array(orig['per_target'])))<1e-12
cache=load('protenix_cache_index');depths=sorted(cache['records'][i]['teacher_depth'] for i in cache['train_target_ids']);assert statistics.median(depths)==cache['fixed_depth']==508.5
split=load('protenix_split384');sets={k:set(split[k]) for k in ['train24_ids','train96_ids','train384_ids','dev8_ids','confirmation_ids','all_observed_confirmation_ids']};length={v['target_id'] for v in load('length48_manifest')['targets']}
assert sets['train24_ids']<=sets['train96_ids']<=sets['train384_ids']
pairchecks={f'{a} vs {b}':sorted(sets[a]&sets[b]) for a,b in [('train384_ids','dev8_ids'),('train384_ids','confirmation_ids'),('train384_ids','all_observed_confirmation_ids')]}
pairchecks['Length48 vs recorded train/dev/old confirmation']=sorted(length&set.union(*sets.values()));assert not any(pairchecks.values())
manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'source':str(SRC/copies[p.name]) if p.name in copies else 'existing evidence snapshot'} for p in E.glob('*.json')}
code_inputs=['src/engramfold/models/interface_heads.py','src/engramfold/models/native_geometry.py','src/engramfold/models/live_opm.py','docs/native_direction96_v1.md','docs/plm_interface_control_v1.md','docs/train384_direction_v1.md']
code_hashes={p:hashlib.sha256((SRC/p).read_bytes()).hexdigest() for p in code_inputs}
result={'code_and_protocol_sha256':code_hashes,'snapshot_utc':datetime.now(timezone.utc).isoformat(),'scope':'existing-score descriptive audit; no new CI, filtering or selection','contrasts':contrasts,'matched_full_1536_interaction':{'mean':float(np.mean(list(interaction.values()))),'original_ci95':orig['ci95'],'per_target':interaction},'depth_audit':{'n_train':len(depths),'sorted_train_depths':depths,'middle_two':depths[11:13],'median':statistics.median(depths),'cache_fixed_depth':cache['fixed_depth']},'exact_identifier_isolation':pairchecks,'exposure_inventory_counts':load('cumulative_exposure')['counts'],'files':manifest}
save('coverage_robustness_audit.json',result)
print('PASS',len(contrasts),'contrasts, all target arrays matched original; D median and exact-ID isolation verified')
for c in contrasts:print(c['label'],round(c['native_mean'],5),round(c['control_mean'],5),'seed',np.round(c['per_seed'],5),'rot',None if c['per_rotation'] is None else np.round(c['per_rotation'],5),'positive',c['positive'],'median',round(c['quantiles']['0.5'],5),'9cells',c.get('positive_seed_rotation_cells'))
