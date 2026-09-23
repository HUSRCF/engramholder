#!/usr/bin/env python3
"""Generate manuscript numbers, tables and figures from frozen evidence JSON.

No training, CIF rescoring, endpoint changes, or automatic input-lock updates.
Use --init-lock exactly once when accepting the current evidence snapshot.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'generated'
FIG = ROOT / 'figures'
SOURCES = ['protenix_direction','protenix_extensions','protenix_train384','length48',
 'openfold_train96','openfold_train384','openfold_paired','openfold_gplus_rotation_final',
 'atlasfold_adapters_final','protenix_data_interaction','protenix_generic',
 'openfold_gplus_rotation_records','openfold_train384_records','atlasfold_adapters_records',
 'protenix_gplus384_summary','protenix_gplus384_records','protenix_gplus384_verification',
 'protenix_gplus384_historical_confirm96_records','length48_records',
 'protenix_gplus384_collection_audit','protenix_gplus384_execution_lock',
 'full_cross_summary','full_cross_records','full_cross_execution_lock','full_cross_panel','full_cross_completion','full_cross_verification','v4_analysis','v4_e3_analysis','v4_independent_verification','v4_source_hash_audit','openfold_esmc_A_summary','openfold_esmc_A_records','openfold_esmc_A_execution_lock','openfold_esmc_A_scoring_lock','openfold_esmc_A_verification','data_composition_audit','openfold_fresh96_summary','openfold_fresh96_records','openfold_fresh96_complete','openfold_fresh96_execution_lock','openfold_fresh96_model_lock','openfold_fresh96_scoring_lock','openfold_fresh96_reference_manifest','openfold_training_psi_protocol','openfold_followup_analysis']
CELLS = {}
DATA = {}

def read(file, path):
    x = DATA[file]
    for key in path: x = x[key]
    return x

def number(key, file, path, operation='identity', signed=False, scientific=False):
    value = read(file, path)
    if operation == 'mean': value = float(np.mean(value))
    value = float(value)
    assert np.isfinite(value)
    CELLS[key] = dict(source=f'evidence/{file}.json', field_path=path,
                      operation=operation, value=value, formatted=(f'{value:+.5e}' if signed else f'{value:.5e}') if scientific else (f'{value:+.5f}' if signed else f'{value:.5f}'))
    return value

def mean_systems(key, file, prefix, names):
    paths = [prefix + [n] for n in names]
    values = [float(read(file, p)) for p in paths]
    value = float(np.mean(values))
    CELLS[key] = dict(source=f'evidence/{file}.json', field_paths=paths,
                     operation='arithmetic mean of complete system means',value=value,formatted=f'{value:.5f}')
    return value

def contrast(key, file, path, scientific=False):
    obj=read(file,path)
    m=number(key,file,path+['mean'],signed=True,scientific=scientific)
    lo=number(key+'Lo',file,path+['ci95',0],signed=True,scientific=scientific)
    hi=number(key+'Hi',file,path+['ci95',1],signed=True,scientific=scientific)
    assert lo<=m<=hi
    if 'per_target' in obj: assert abs(np.mean(obj['per_target'])-m)<1e-12
    return (m,lo,hi)

def fmt(key): return CELLS[key]['formatted']
def tex(key): return r'\result{'+key+'}'
def names_matching(file,path,pattern,expected):
    names=[n for n in read(file,path) if re.fullmatch(pattern,n)]
    assert len(names)==expected,(file,pattern,len(names))
    return names

def verify_full_cross():
    rows=DATA['full_cross_records'];summary=DATA['full_cross_summary']
    lock=DATA['full_cross_execution_lock'];targets=DATA['full_cross_panel']['targets']
    assert len(rows)==720 and all(x['status']=='ok' for x in rows)
    assert len(targets)==24 and len(lock['seeds'])==len(lock['rotations'])==3
    assert DATA['full_cross_completion']['passed'] and DATA['full_cross_verification']['passed']
    ids=[x['target_id'] for x in targets];rng=np.random.default_rng(lock['bootstrap_seed'])
    draw=np.concatenate([rng.choice([i for i,t in enumerate(targets) if t['length_bin']==b],size=(lock['bootstrap_samples'],6)) for b in ['128-191','192-255','256-319','320-384']],axis=1)
    checked=0
    for metric in ['ca_lddt','residue_ca_lddt','tm_score']:
        by={(x['system'],x['target_id']):x[metric] for x in rows};assert len(by)==720
        arrays={k:np.empty((3,3,24)) for k in ['NN','NR','RN','RR','D_mN','D_mR','A_dN','A_dR','Edir','Eamp','I','NN_minus_RR']}
        for si,seed in enumerate(lock['seeds']):
            for ri,rot in enumerate(lock['rotations']):
                for ti,target in enumerate(ids):
                    nn=by[f'NN_s{seed}',target];nr=by[f'NR_r{rot}_s{seed}',target]
                    rn=by[f'RN_r{rot}_s{seed}',target];rr=by[f'RR_r{rot}_s{seed}',target]
                    values=dict(NN=nn,NR=nr,RN=rn,RR=rr,D_mN=nn-rn,D_mR=nr-rr,A_dN=nn-nr,A_dR=rn-rr,Edir=(nn-rn+nr-rr)/2,Eamp=(nn-nr+rn-rr)/2,I=nn-rn-nr+rr,NN_minus_RR=nn-rr)
                    for k,v in values.items():arrays[k][si,ri,ti]=v
        for k,v in arrays.items():
            reported=summary['metrics'][metric]['cells' if k in ['NN','NR','RN','RR'] else 'contrasts'][k]
            target=v.mean((0,1));calc=dict(mean=target.mean(),per_target=target,per_seed=v.mean((1,2)),per_rotation=v.mean((0,2)),ci95=np.quantile(target[draw].mean(1),[.025,.975]))
            for name,value in calc.items():assert np.max(abs(np.asarray(value)-reported[name]))<1e-12,(metric,k,name)
            assert int((target>0).sum())==reported['positive_targets'];checked+=1
        assert np.max(abs(arrays['Edir']+arrays['Eamp']-arrays['NN_minus_RR']))<1e-12
    return checked

def validate_numeric_keys(root, cells):
    """Check literal result references in current manuscript and generated tables."""
    files=[root/'iclr2027_conference.tex']
    for folder in ['sections','appendices','generated']:
        files.extend(sorted((root/folder).glob('*.tex')))
    uses={}
    for path in files:
        for line,text in enumerate(path.read_text().splitlines(),1):
            text=re.sub(r'(?<!\\)%.*','',text)
            for key in re.findall(r'\\result\{([^{}]+)\}',text):
                uses.setdefault(key,[]).append(f'{path.relative_to(root)}:{line}')
    if not uses:raise ValueError('No literal numerical references found in the manuscript')
    missing={key:locations for key,locations in uses.items() if key not in cells}
    if missing:raise ValueError('Missing numerical keys: '+json.dumps(missing,sort_keys=True))
    return {'literal_references':sum(map(len,uses.values())), 'distinct_keys':len(uses), 'missing':[]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--init-lock',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    hashes={f'evidence/{f}.json':hashlib.sha256((ROOT/'evidence'/f'{f}.json').read_bytes()).hexdigest() for f in SOURCES}
    lock=ROOT/'notes/writing_branch_20260922/paper_sources.v6.lock.json'
    if args.init_lock:
        if lock.exists(): raise FileExistsError('Input lock exists; do not overwrite')
        lock.write_text(json.dumps(hashes,indent=2)+'\n')
    assert json.loads(lock.read_text())==hashes,'Evidence changed; review it before creating a new versioned lock'
    DATA.update({f:json.loads((ROOT/'evidence'/f'{f}.json').read_text()) for f in SOURCES})
    from verify_openfold_esmc_A import verify
    esmc_audit=verify(ROOT)
    from analyze_openfold_followups import calculate
    followup_audit=calculate(ROOT)
    assert followup_audit==DATA['openfold_followup_analysis']
    # Recheck completed raw-score system means without folding or changing statistics.
    checks=0
    for file, records_file in [('openfold_gplus_rotation_final','openfold_gplus_rotation_records'),('openfold_train384','openfold_train384_records'),('atlasfold_adapters_final','atlasfold_adapters_records')]:
        records=DATA[records_file]
        for panel,pd in DATA[file]['panels'].items():
            for metric,md in pd.items():
                for name,value in md['system_means'].items():
                    vals=[r[metric] for r in records if r['panel']==panel and r['system']==name]
                    # Atlas unadapted baseline is in its separate system snapshot.
                    if not vals and file=='atlasfold_adapters_final' and name=='query_native_plm': continue
                    assert vals and len(vals)==(96 if panel=='confirm96' else 48),(file,panel,name)
                    assert abs(np.mean(vals)-value)<1e-12,(file,panel,metric,name)
                    checks+=1
    # Check the core contrast at each target, not only aggregate system means.
    interaction_checks=0
    raw=DATA['openfold_gplus_rotation_records']
    for panel in ['confirm96','length48']:
        md=DATA['openfold_gplus_rotation_final']['panels'][panel]['ca_lddt']
        for target,expected in zip(md['interaction']['target_ids'],md['interaction']['per_target'],strict=True):
            by={r['system']:r['ca_lddt'] for r in raw if r['panel']==panel and r['target_id']==target}
            def avg(pattern,n):
                vals=[v for k,v in by.items() if re.fullmatch(pattern,k)]
                assert len(vals)==n
                return np.mean(vals)
            observed=avg('native_s[0-9]+',3)-avg('r[0-9]+_s[0-9]+',9)-avg('gplus_s[0-9]+',3)+avg('gplus_r[0-9]+_s[0-9]+',9)
            assert abs(observed-expected)<1e-12
            interaction_checks+=1
    # New matched Protenix G+ scores: independently reconstruct six locked contrasts.
    pt_checks=0
    for panel in ['confirm96','length48']:
        pd=DATA['protenix_gplus384_summary'][panel]
        assert pd['failure_count']==0
        ids=pd['target_ids'];assert len(ids)==len(set(ids))==(96 if panel=='confirm96' else 48)
        old=DATA['protenix_gplus384_historical_confirm96_records'] if panel=='confirm96' else DATA['length48_records']
        new=DATA['protenix_gplus384_records'][panel]
        for metric,md in pd['metrics'].items():
            oldmetric=({'ca_pair_lddt':'original_ca_lddt','residue_ca_lddt':'residue_average_ca_lddt'}.get(metric,metric) if panel=='confirm96' else metric)
            def arr(rows,key):
                by={x['target_id']:x for x in rows};assert len(rows)==len(by)==len(ids) and set(by)==set(ids)
                return np.array([by[i][key] for i in ids])
            n=np.stack([arr(old[f'n384_full_native_s{s}_u1536'],oldmetric) for s in [20260923,20260924,20260925]])
            g=np.stack([arr(new[f'gplus_s{s}'],metric) for s in [20260923,20260924,20260925]])
            assert abs(g.mean()-md['means']['generic_plus'])<1e-12
            assert np.max(abs(g.mean(1)-md['gplus_per_seed']))<1e-12
            target=(n-g).mean(0);reported=md['native_minus_gplus']
            assert np.max(abs(target-reported['per_target']))<1e-12
            draws=np.random.default_rng(20260926).integers(len(ids),size=(20000,len(ids)))
            ci=np.quantile(target[draws].mean(1),[.025,.975])
            assert np.max(abs(ci-reported['ci95']))<1e-12
            assert abs(target.mean()-reported['mean'])<1e-12
            assert (target>0).sum()==reported['positive_targets']
            pt_checks+=1
    assert DATA['protenix_gplus384_verification']['passed'] and DATA['protenix_gplus384_collection_audit']['passed']
    for panel,short in [('confirm96','c96'),('length48','l48')]:
        pre=[panel,'metrics']
        number(f'pt384_{short}_gplus','protenix_gplus384_summary',pre+['ca_pair_lddt','means','generic_plus'])
        number(f'pt384_{short}_gquery','protenix_gplus384_summary',pre+['ca_pair_lddt','gplus_minus_query','mean'],signed=True)
        number(f'pt384_{short}_gmedian','protenix_gplus384_verification',['results',panel,'ca_pair_lddt','median'],signed=True)
        for metric,suffix in [('ca_pair_lddt','gdiff'),('residue_ca_lddt','gdiff_residue'),('tm_score_fixed_full_length','gdiff_tm')]:
            contrast(f'pt384_{short}_{suffix}','protenix_gplus384_summary',pre+[metric,'native_minus_gplus'])
    rows=[]
    for size,file in [(96,'protenix_direction'),(384,'protenix_train384')]:
        pre=['systems']; means={n:v['mean_ca_lddt'] for n,v in read(file,pre).items()}
        keys=[]
        for group,pattern,num in [('native',f'n{size}_full_native_s[0-9]+_u1536',3),('rotated',f'n{size}_full_r[0-9]+_s[0-9]+_u1536',9)]:
            source_file=file
            group_means=means
            if size==96 and group=='rotated':
                source_file='protenix_extensions';pattern='C_mini_ordinary_r[0-9]+_s[0-9]+'
                group_means={n:v['mean_ca_lddt'] for n,v in DATA[source_file]['systems'].items()}
            ns=[n for n in group_means if re.fullmatch(pattern,n)];assert len(ns)==num,(source_file,pattern,len(ns))
            key=f'pt{size}_c96_{group}';value=np.mean([group_means[n] for n in ns]);CELLS[key]=dict(source=f'evidence/{source_file}.json',field_paths=[['systems',n,'mean_ca_lddt'] for n in ns],operation='mean',value=float(value),formatted=f'{value:.5f}');keys.append(key)
        number(f'pt{size}_c96_query','protenix_direction',['systems','query','mean_ca_lddt'])
        rows.append((f'Protenix, Train{size}', 'C96-B', [f'pt{size}_c96_query',*keys], r'---' if size==96 else tex('pt384_c96_gplus')))
    for group,pattern,n in [('native','n384_full_native_s[0-9]+_u1536',3),('rotated','n384_full_r[0-9]+_s[0-9]+_u1536',9)]:
        pre=['metrics','ca_pair_lddt','absolute_means'];ns=names_matching('length48',pre,pattern,n);mean_systems('pt384_l48_'+group,'length48',pre,ns)
    number('pt384_l48_query','length48',['metrics','ca_pair_lddt','absolute_means','query'])
    number('pt384_l48_official','length48',['metrics','ca_pair_lddt','absolute_means','official_mini_esm'])
    rows.append(('Protenix, Train384','L48',['pt384_l48_query','pt384_l48_native','pt384_l48_rotated'],tex('pt384_l48_gplus')))
    for size in [96,384]:
        file=f'openfold_train{size}'
        for panel,short in [('confirm96','c96'),('length48','l48')]:
            pre=['panels',panel,'ca_lddt','system_means'];keys=[]
            for group,pattern,n in [('native','native_s[0-9]+',3),('rotated','r[0-9]+_s[0-9]+',9),('gplus','gplus_s[0-9]+',3)]:
                key=f'of{size}_{short}_{group}';mean_systems(key,file,pre,names_matching(file,pre,pattern,n));keys.append(key)
            number(f'of{size}_{short}_query',file,pre+['query'])
            rows.append((f'OpenFold, Train{size}','C96-B' if short=='c96' else 'L48',[f'of{size}_{short}_query',*keys[:-1]],tex(keys[-1])))
            contrast(f'of{size}_{short}_direction',file,['panels',panel,'ca_lddt','native_minus_rotated'])
            contrast(f'of{size}_{short}_gdiff','openfold_paired',['panels',panel,'ca_lddt',f'train{size}_native_minus_gplus'])
    for panel,short in [('confirm96','c96'),('length48','l48')]:
        pre=['panels',panel,'ca_lddt']
        for g in ['query_native_plm','native','rotated','gplus']: number('atlas_'+short+'_'+g,'atlasfold_adapters_final',pre+['means',g])
        rows.append(('AtlasFold, Train96','C96-B' if short=='c96' else 'L48',[f'atlas_{short}_query_native_plm',f'atlas_{short}_native',f'atlas_{short}_rotated'],tex(f'atlas_{short}_gplus')))
        contrast('atlas_'+short+'_direction','atlasfold_adapters_final',pre+['native_minus_rotated'])
        contrast('atlas_'+short+'_gdiff','atlasfold_adapters_final',pre+['native_minus_gplus'])
        for g in ['factor_native','factor_rotated','gplus_native','gplus_rotated']:
            number(f'inter_{short}_{g}','openfold_gplus_rotation_final',pre+['group_means',g])
        for kind in ['factor_rotation','gplus_rotation','interaction']:
            contrast(f'inter_{short}_{kind}','openfold_gplus_rotation_final',pre+[kind])
        contrast('of_scale_'+short,'openfold_paired',pre+['direction_interaction_384_minus_96'])
    contrast('pt_tangent','protenix_direction',['primary'])
    contrast('pt_full24','protenix_direction',['secondary','native_full_minus_rotated_full_n24_u384'])
    contrast('pt_tiny','protenix_extensions',['primary'])
    contrast('pt_mean','protenix_extensions',['secondary','mini_native_minus_mean_preserving'])
    contrast('pt_full96','protenix_extensions',['secondary','mini_train96_full_native_minus_rotated'])
    contrast('pt_full384','protenix_train384',['primary'])
    contrast('pt_length','length48',['metrics','ca_pair_lddt','primary'])
    contrast('pt_scale','protenix_data_interaction',['direction_difference_in_differences'])
    contrast('pt_scale_native','protenix_data_interaction',['native_train384_minus96'])
    cross_checks=verify_full_cross()
    for cell in ['NN','NR','RN','RR']:
        number('cross_'+cell,'full_cross_summary',['metrics','ca_lddt','cells',cell,'mean'])
    for metric,short in [('ca_lddt','ca'),('residue_ca_lddt','res'),('tm_score','tm')]:
        for key in ['D_mN','D_mR','Edir','Eamp','I','A_dN','A_dR','NN_minus_RR']:
            contrast(f'cross_{short}_{key}','full_cross_summary',['metrics',metric,'contrasts',key])
    for name,field,sci in [('oracle','oracle_D_difference',False),('equal','dynamic_equal_decrease_difference',True),('actual','dynamic_actual_loss_decrease_difference',False),('norm_diff','norm_ratio_difference',False)]:
        contrast('v4_'+name,'v4_analysis',['metrics',field],scientific=sci)
    for key in ['norm_ratio_native','norm_ratio_rotated']:
        number('v4_'+key,'v4_analysis',['metrics',key,'mean'])
    contrast('v4_retain','v4_e3_analysis',['metrics','retaining_effect_difference'])
    assert DATA['v4_independent_verification']['passed']
    for file in ['v4_analysis','v4_e3_analysis']:
        d=DATA[file]
        for key,m in d['metrics'].items():
            assert abs(np.mean([row[key] for row in d['chains']])-m['mean'])<1e-12
    def interval_cell(k):return r'$['+tex(k+'Lo')+', '+tex(k+'Hi')+']$'
    def save_rows(name,rows):
        (OUT/name).write_text('\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+r'\bottomrule'+'\n')
    save_rows('cross_cells.tex', [['Native source',tex('cross_NN'),tex('cross_NR')],['Rotated source',tex('cross_RN'),tex('cross_RR')]])
    save_rows('cross_effects.tex',[[label,tex('cross_ca_'+key),interval_cell('cross_ca_'+key)] for label,key in [('At native norm','D_mN'),('At rotated norm','D_mR'),('Average source effect','Edir')]])
    save_rows('cross_supplement.tex',[[label,key.replace('_',r'\_'),tex('cross_'+short+'_'+key),interval_cell('cross_'+short+'_'+key)] for label,short in [('Pair-lDDT','ca'),('Residue-lDDT','res'),('TM-score','tm')] for key in ['Edir','Eamp','I','A_dN','A_dR']])
    save_rows('v4_mechanism_rows.tex',[[label,tex('v4_'+key),interval_cell('v4_'+key)] for label,key in [('E1 oracle (normalized)','oracle'),('E2 learned, small equal norm (normalized)','equal'),('E2 learned, actual norm (raw loss)','actual'),('E3 final-residual retention (raw loss)','retain')]])
    a_rows=[];a_contrasts=[]
    for panel,short,label in [('confirm96','c96','C96-B'),('length48','l48','L48')]:
        for feat,tag,flabel in [('E_last','e','ESM2-35M'),('C_last','c','ESMC-600M')]:
            pre=['panels',panel,'ca_lddt','cells',feat];prefix=f'a_{short}_{tag}_'
            keys=['query','native','rotated_factor','gplus','rotated_gplus']
            for key in keys:number(prefix+key,'openfold_esmc_A_summary',pre+['means',key])
            a_rows.append([label,flabel,*[tex(prefix+k) for k in keys]])
            for metric,mlabel in [('ca_lddt','Pair-lDDT'),('residue_ca_lddt','Residue-lDDT'),('tm_score_fixed_full_length','TM-score')]:
                for k,kl in [('factor_rotation',r'$\Delta_F$'),('gplus_rotation',r'$\Delta_{G+}$'),('interaction',r'$\Psi$'),('native_minus_gplus','Native--G+')]:
                    key=prefix+metric+'_'+k
                    contrast(key,'openfold_esmc_A_summary',['panels',panel,metric,'cells',feat,k])
                    a_contrasts.append([label,flabel,mlabel,kl,tex(key),interval_cell(key)])
        for k in ['direction_change','psi_change','native_gain','gplus_gain','native_gplus_gap_change']:
            contrast(f'a_{short}_{k}','openfold_esmc_A_summary',['panels',panel,'ca_lddt','plm_interactions',k])
    for k in ['K','Psi_C','J']:number('a_holm_'+k,'openfold_esmc_A_summary',['key_secondary_holm_p',k])
    save_rows('esmc_A_means.tex',a_rows)
    scope_interactions=[]
    for label,keys in [('Train96 / ESM2',['inter_c96_interaction','inter_l48_interaction']),('Train384 / ESM2',['a_c96_e_ca_lddt_interaction','a_l48_e_ca_lddt_interaction']),('Train384 / ESMC',['a_c96_c_ca_lddt_interaction','a_l48_c_ca_lddt_interaction'])]:
        scope_interactions.append([label,*[tex(k)+' '+interval_cell(k) for k in keys]])
    save_rows('interaction_scope_rows.tex',scope_interactions)

    fresh_rows=[]; training_rows=[]
    for metric,label in [('ca_lddt','Pair-lDDT'),('residue_ca_lddt','Residue-lDDT'),('tm_score_fixed_full_length','TM-score')]:
        for k in ['factor_rotation','gplus_rotation','interaction','native_minus_gplus','factor_minus_query','generic_plus_minus_query']:
            key=f'fresh_{metric}_{k}'
            contrast(key,'openfold_fresh96_summary',['metrics',metric,k])
            fresh_rows.append([label,k.replace('_',r'\_'),tex(key),interval_cell(key)])
        for panel,short,label_panel in [('confirm96','c96','C96-B'),('length48','l48','L48')]:
            key=f'psi_train_{short}_{metric}'
            contrast(key,'openfold_followup_analysis',['training_setting_change','panels',panel,metric])
            training_rows.append([label_panel,label,tex(key),interval_cell(key)])
    for k in ['factor_native','factor_rotated','generic_plus_native','generic_plus_rotated','query_native']:
        number('fresh_'+k,'openfold_fresh96_summary',['metrics','ca_lddt','group_means',k])
    marginal_rows=[]
    for kind in ['seed','rotation']:
        keys=[]
        for i in range(3):
            key=f'fresh_{kind}_{i}'
            number(key,'openfold_fresh96_summary',['metrics','ca_lddt','interaction_by_'+kind,i],signed=True)
            keys.append(tex(key))
        marginal_rows.append([kind.capitalize(),*keys])
    save_rows('fresh96_marginals.tex',marginal_rows)
    save_rows('fresh96_contrasts.tex',fresh_rows)
    save_rows('training_psi_change.tex',training_rows)

    # One compact supplemental table per metric avoids an unbreakable 48-row float.
    for metric,label in [('pair','Pair-lDDT'),('residue','Residue-lDDT'),('tm','TM-score')]:
        save_rows('esmc_A_'+metric+'_contrasts.tex',[[*x[:2],*x[3:]] for x in a_contrasts if x[2]==label])
    for metric,suffix in [('residue_ca_lddt','residue'),('tm_score_fixed_full_length','tm')]:
        contrast('inter_c96_interaction_'+suffix,'openfold_gplus_rotation_final',['panels','confirm96',metric,'interaction'])
    data_audit=DATA['data_composition_audit'];assert data_audit['passed'] and data_audit['nested']
    groups=data_audit['datasets']
    for name,d in groups.items():
        assert d['count']==len(d['records'])==len({x['target_id'] for x in d['records']})
        assert d['actual_length']==[min(x['length'] for x in d['records']),max(x['length'] for x in d['records'])]
    sets={k:{x['target_id'] for x in v['records']} for k,v in groups.items()}
    assert sets['Train24']<sets['Train96']<sets['Train384']
    assert all(not sets['Train384']&sets[k] for k in ['Dev8','Confirm96-A','Confirm96-B','Length48'])
    data_specs=[
        ('Train24','Inherited Train24','Original cache; no new cutoff','Subset of Train96','Adaptation training'),
        ('Dev8','Inherited Dev8','Original cache; no new cutoff','Disjoint from training','Development; already observed'),
        ('Train96-additions','Train96 additions','Short-chain catalog (Q)','18/stratum added to Train24; excludes B','Adaptation training; B selected first'),
        ('Train384-additions','Train384 additions','Short-chain catalog (Q)','72/stratum added to Train96; excludes Dev8/A/B','Adaptation training'),
        ('Confirm96-A','Confirm96-A','Short-chain catalog (Q)','Hash order; excludes old queries/MSA and accepted neighbors','New for Factor--Generic'),
        ('Confirm96-B','Confirm96-B','Short-chain catalog (Q)','Hash order; excludes A, old queries/MSA and accepted neighbors','New for direction study'),
        ('Length48','Length48','Long-chain catalog; v2 rules below','Excludes recorded training/development and A/B','New for Protenix length transfer')]
    table=[]
    for name,label,source,exclusions,use in data_specs:
        d=groups[name];counts=list(d['bins'].values());assert len(set(counts))==1
        lo,hi=d['actual_length'];comp=f"{len(counts)} x {counts[0]}; {lo}--{hi}"
        table.append([label,str(d['count']),source,comp,exclusions,use])
    fresh_targets=DATA['openfold_fresh96_reference_manifest']['targets']
    assert len(fresh_targets)==96 and len({t['target_id'] for t in fresh_targets})==96
    table.append(['Fresh96','96','Short-chain eligibility (Q); BLAST v2',
                  '4 x 24; 129--384','Excludes recorded exposure and accepted neighbors',
                  'New for fixed OpenFold Train96 interaction'])
    save_rows('data_composition_rows.tex',table)
    # All text/table numerical macros derive from these same sources.
    (OUT/'numbers.tex').write_text('% Generated; edit sources/script, not numbers.\n'+''.join(r'\expandafter\def\csname data:'+k+r'\endcsname{'+v['formatted']+'}\n' for k,v in CELLS.items()))
    (OUT/'main_table_rows.tex').write_text('% Generated from fixed source hashes.\n'+'\n'.join(' & '.join([model,panel,*[tex(k) for k in keys],g])+r' \\' for model,panel,keys,g in rows)+'\n')
    (OUT/'paired_table_rows.tex').write_text('\n'.join(' & '.join([model,panel,tex(key),r'$['+tex(key+'Lo')+', '+tex(key+'Hi')+']$'])+r' \\' for model,panel,key in [('Protenix, Train384','C96-B','pt384_c96_gdiff'),('Protenix, Train384','L48','pt384_l48_gdiff'),('OpenFold, Train96','C96-B','of96_c96_gdiff'),('OpenFold, Train96','L48','of96_l48_gdiff'),('OpenFold, Train384','C96-B','of384_c96_gdiff'),('OpenFold, Train384','L48','of384_l48_gdiff'),('AtlasFold, Train96','C96-B','atlas_c96_gdiff'),('AtlasFold, Train96','L48','atlas_l48_gdiff')])+'\n')
    (OUT/'protenix_gplus_supplement_rows.tex').write_text('\n'.join(' & '.join([panel,label,tex(key),r'$['+tex(key+'Lo')+', '+tex(key+'Hi')+']$'])+r' \\' for panel,short in [('C96-B','c96'),('L48','l48')] for label,suffix in [('Pair-lDDT','gdiff'),('Residue-lDDT','gdiff_residue'),('TM-score','gdiff_tm')] for key in [f'pt384_{short}_{suffix}'])+'\n'+r'\bottomrule'+'\n')
    scope=[('Protenix Mini: Train24, tangent, 384 / C96-B','pt_tangent','P'),('Protenix Mini: Train24, Full, 384 / C96-B','pt_full24','S'),('Protenix Tiny: Train24, tangent, 384 / C96-B','pt_tiny','F'),('Protenix Mini: mean-preserving R / C96-B','pt_mean','F'),('Protenix Mini: Train96, Full, 1536 / C96-B','pt_full96','F'),('Protenix Mini: Train384, Full, 1536 / C96-B','pt_full384','F'),('Protenix Mini: Train384, Full, 1536 / L48','pt_length','P'),('OpenFold ESM2: Train96, Full, 1536 / C96-B','of96_c96_direction','F'),('OpenFold ESM2: Train96, Full, 1536 / L48','of96_l48_direction','F'),('OpenFold ESM2: Train384, Full, 1536 / C96-B','of384_c96_direction','F'),('OpenFold ESM2: Train384, Full, 1536 / L48','of384_l48_direction','F'),('AtlasFold: Train96, 1536 / C96-B','atlas_c96_direction','F'),('AtlasFold: Train96, 1536 / L48','atlas_l48_direction','F')]
    scope[11:11]=[('OpenFold ESMC: Train384, Full, 1536 / C96-B','a_c96_c_ca_lddt_factor_rotation','F'),('OpenFold ESMC: Train384, Full, 1536 / L48','a_l48_c_ca_lddt_factor_rotation','F')]
    (OUT/'scope_table_rows.tex').write_text('\n'.join(' & '.join([label.replace('_',r'\_'),status,tex(key),r'$['+tex(key+'Lo')+', '+tex(key+'Hi')+']$'])+r' \\' for label,key,status in scope)+'\n')
    for table in ["main_table_rows", "paired_table_rows", "scope_table_rows"]:
        p=OUT/(table+".tex");p.write_text(p.read_text()+r"\bottomrule"+"\n")
    key_audit=validate_numeric_keys(ROOT,CELLS)
    draw_figures(scope)
    (OUT/'cell_sources.json').write_text(json.dumps({'inputs':hashes,'cells':CELLS,'verified_esmc_A_contrasts':esmc_audit['verified_contrasts'],'verified_raw_system_metric_means':checks,'verified_target_interactions':interaction_checks,'verified_new_protenix_contrasts':pt_checks,'verified_full_cross_cells_and_contrasts':cross_checks,'numeric_key_audit':key_audit,'pending':[],'unrun':['pt96_c96_gplus']},indent=2)+'\n')
    print(f'Generated {len(CELLS)} numeric fields; checked {checks} raw-score system/metric means.')

def draw_figures(scope):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    def val(k):return CELLS[k]['value']
    # Figure 2: absolute means are descriptive, paired intervals are separate.
    fig,ax=plt.subplots(2,2,figsize=(6.6,3.35),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained')
    cols=['#245a81','#799cb5','#9b5d16','#d4af7b']
    for j,(s,title) in enumerate([('c96','Train96 / ESM2: Confirm96-B'),('l48','Train96 / ESM2: Length48')]):
        groups=['factor_native','factor_rotated','gplus_native','gplus_rotated']
        vs=[val(f'inter_{s}_{g}') for g in groups]
        ax[j,0].bar(range(4),vs,color=cols,width=.65)
        for i,v in enumerate(vs):ax[j,0].text(i,v+.009,f'{v:.3f}',ha='center',fontsize=9)
        ax[j,0].tick_params(labelsize=9)
        ax[j,0].set_title(title,fontsize=9)
        ax[j,0].set(xticks=range(4),xticklabels=['F','RF','G+','RG+'],ylim=(0,.57),ylabel='Mean pair-lDDT')
        for i,(k,label) in enumerate([('factor_rotation',r'$\Delta_F$'),('gplus_rotation',r'$\Delta_{G+}$'),('interaction',r'$\Psi$')]):
            key=f'inter_{s}_{k}';m,lo,hi=val(key),val(key+'Lo'),val(key+'Hi')
            ax[j,1].errorbar(m,2-i,xerr=[[m-lo],[hi-m]],fmt='o' if i<2 else 'D',color='#333333' if i<2 else '#245a81',capsize=3)
            ax[j,1].text(.041,2-i,f'{m:+.5f}',va='center',fontsize=9)
        ax[j,1].axvline(0,color='.6',lw=.7);ax[j,1].set(yticks=[2,1,0],yticklabels=[r'$\Delta_F$',r'$\Delta_{G+}$',r'$\Psi$'],ylim=(-.65,2.65),xlim=(-.014,.057),xticks=[-.01,0,.01,.02,.03],xlabel='Paired difference (95% CI)')
        ax[j,1].tick_params(labelsize=9)
    fig.savefig(FIG/'interaction.pdf');fig.savefig(FIG/'interaction.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,4.65));fig.subplots_adjust(left=.52,right=.98,top=.98,bottom=.12)
    for i,(label,k,status) in enumerate(scope):
        y=len(scope)-1-i;m,lo,hi=val(k),val(k+'Lo'),val(k+'Hi')
        color='#245a81' if label.startswith('Protenix') else '#9b5d16' if label.startswith('OpenFold') else '#587b46'
        ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt={'P':'o','S':'^','F':'s'}[status],color=color,capsize=2,markersize=4)
    ax.axvline(0,color='.5',lw=.8);ax.set(yticks=range(len(scope)),yticklabels=[f'[{s}] {l.replace("Protenix Mini:","Mini:").replace("Protenix Tiny:","Tiny:").replace("Train","n=")}' for l,k,s in reversed(scope)],xlabel='Native minus rotated (pair-lDDT)',xlim=(-.012,.13));ax.tick_params(axis='y',labelsize=9)
    fig.savefig(FIG/'scope.pdf');fig.savefig(FIG/'scope.png',dpi=200);plt.close(fig)
    # Figure 1: standard vector drawing of the actual experimental design.
    fig,ax=plt.subplots(figsize=(6.6,2.65));fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.01);ax.set(xlim=(0,10),ylim=(0,4.3));ax.axis('off')
    def box(x,y,w,h,text,color='#f0f3f5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',fc=color,ec='.4',lw=.7));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(x,y,xx,yy):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=10,lw=.8,color='.3'))
    box(.1,2.6,1.5,1,'Query\nFrozen PLM');box(2,2.6,1.65,1,'Increment head\nTRAINABLE','#e1edf7');box(4.1,2.6,2.1,1,'Native operator\nresidual\nFROZEN');box(6.65,2.6,1.1,1,r'$R\Delta U$'+'\nR fixed');box(8.2,2.6,1.6,1,'Add baseline\nFrozen folding')
    for x,xx in [(1.6,2),(3.65,4.1),(6.2,6.65),(7.75,8.2)]:arrow(x,3.1,xx,3.1)
    ax.text(5,4,'Factor path: final increment layer starts at zero',ha='center',fontsize=8)
    box(.1,.35,4.1,1.5,'G+: same extra features + query factors\nTrainable pair MLP; free affine output\nZero final layer; fixed residual rotation','#fff3e4')
    ax.text(7.2,2.05,'All four conditions receive task training',ha='center',fontsize=8)
    tbl=ax.table(cellText=[['Factor','Native F','Rotated RF'],['G+','Native G+','Rotated RG+']],colLabels=['Head','R = I','Fixed R'],cellLoc='center',bbox=[.48,.09,.49,.33]);tbl.auto_set_font_size(False);tbl.set_fontsize(9)
    fig.savefig(FIG/'construction.pdf',bbox_inches='tight');fig.savefig(FIG/'construction.png',dpi=200,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
