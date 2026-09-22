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
 'openfold_gplus_rotation_records','openfold_train384_records','atlasfold_adapters_records']
CELLS = {}
DATA = {}

def read(file, path):
    x = DATA[file]
    for key in path: x = x[key]
    return x

def number(key, file, path, operation='identity', signed=False):
    value = read(file, path)
    if operation == 'mean': value = float(np.mean(value))
    value = float(value)
    assert np.isfinite(value)
    CELLS[key] = dict(source=f'evidence/{file}.json', field_path=path,
                      operation=operation, value=value, formatted=f'{value:+.5f}' if signed else f'{value:.5f}')
    return value

def mean_systems(key, file, prefix, names):
    paths = [prefix + [n] for n in names]
    values = [float(read(file, p)) for p in paths]
    value = float(np.mean(values))
    CELLS[key] = dict(source=f'evidence/{file}.json', field_paths=paths,
                     operation='arithmetic mean of complete system means',value=value,formatted=f'{value:.5f}')
    return value

def contrast(key, file, path):
    obj=read(file,path)
    m=number(key,file,path+['mean'],signed=True)
    lo=number(key+'Lo',file,path+['ci95',0],signed=True)
    hi=number(key+'Hi',file,path+['ci95',1],signed=True)
    assert lo<=m<=hi
    if 'per_target' in obj: assert abs(np.mean(obj['per_target'])-m)<1e-12
    return (m,lo,hi)

def fmt(key): return CELLS[key]['formatted']
def tex(key): return r'\result{'+key+'}'
def names_matching(file,path,pattern,expected):
    names=[n for n in read(file,path) if re.fullmatch(pattern,n)]
    assert len(names)==expected,(file,pattern,len(names))
    return names

def main():
    p=argparse.ArgumentParser();p.add_argument('--init-lock',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    hashes={f'evidence/{f}.json':hashlib.sha256((ROOT/'evidence'/f'{f}.json').read_bytes()).hexdigest() for f in SOURCES}
    lock=ROOT/'notes/writing_branch_20260922/paper_sources.lock.json'
    if args.init_lock:
        if lock.exists(): raise FileExistsError('Input lock exists; do not overwrite')
        lock.write_text(json.dumps(hashes,indent=2)+'\n')
    assert json.loads(lock.read_text())==hashes,'Evidence changed; review it before creating a new versioned lock'
    DATA.update({f:json.loads((ROOT/'evidence'/f'{f}.json').read_text()) for f in SOURCES})
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
        rows.append((f'Protenix, Train{size}', 'C96-B', [f'pt{size}_c96_query',*keys], r'---' if size==96 else r'\pending'))
    for group,pattern,n in [('native','n384_full_native_s[0-9]+_u1536',3),('rotated','n384_full_r[0-9]+_s[0-9]+_u1536',9)]:
        pre=['metrics','ca_pair_lddt','absolute_means'];ns=names_matching('length48',pre,pattern,n);mean_systems('pt384_l48_'+group,'length48',pre,ns)
    number('pt384_l48_query','length48',['metrics','ca_pair_lddt','absolute_means','query'])
    number('pt384_l48_official','length48',['metrics','ca_pair_lddt','absolute_means','official_mini_esm'])
    rows.append(('Protenix, Train384','L48',['pt384_l48_query','pt384_l48_native','pt384_l48_rotated'],r'\pending'))
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
    # All text/table numerical macros derive from these same sources.
    (OUT/'numbers.tex').write_text('% Generated; edit sources/script, not numbers.\n'+''.join(r'\expandafter\def\csname data:'+k+r'\endcsname{'+v['formatted']+'}\n' for k,v in CELLS.items()))
    (OUT/'main_table_rows.tex').write_text('% Generated from fixed source hashes.\n'+'\n'.join(' & '.join([model,panel,*[tex(k) for k in keys],g])+r' \\' for model,panel,keys,g in rows)+'\n')
    (OUT/'paired_table_rows.tex').write_text('\n'.join(' & '.join([model,panel,tex(key),r'$['+tex(key+'Lo')+', '+tex(key+'Hi')+']$'])+r' \\' for model,panel,key in [('OpenFold, Train96','C96-B','of96_c96_gdiff'),('OpenFold, Train96','L48','of96_l48_gdiff'),('OpenFold, Train384','C96-B','of384_c96_gdiff'),('OpenFold, Train384','L48','of384_l48_gdiff'),('AtlasFold, Train96','C96-B','atlas_c96_gdiff'),('AtlasFold, Train96','L48','atlas_l48_gdiff')])+'\n')
    scope=[('Protenix Mini: Train24, tangent, 384 / C96-B','pt_tangent','P'),('Protenix Mini: Train24, Full, 384 / C96-B','pt_full24','S'),('Protenix Tiny: Train24, tangent, 384 / C96-B','pt_tiny','F'),('Protenix Mini: mean-preserving R / C96-B','pt_mean','F'),('Protenix Mini: Train96, Full, 1536 / C96-B','pt_full96','F'),('Protenix Mini: Train384, Full, 1536 / C96-B','pt_full384','F'),('Protenix Mini: Train384, Full, 1536 / L48','pt_length','P'),('OpenFold: Train96, Full, 1536 / C96-B','of96_c96_direction','F'),('OpenFold: Train96, Full, 1536 / L48','of96_l48_direction','F'),('OpenFold: Train384, Full, 1536 / C96-B','of384_c96_direction','F'),('OpenFold: Train384, Full, 1536 / L48','of384_l48_direction','F'),('AtlasFold: Train96, 1536 / C96-B','atlas_c96_direction','F'),('AtlasFold: Train96, 1536 / L48','atlas_l48_direction','F')]
    (OUT/'scope_table_rows.tex').write_text('\n'.join(' & '.join([label.replace('_',r'\_'),status,tex(key),r'$['+tex(key+'Lo')+', '+tex(key+'Hi')+']$'])+r' \\' for label,key,status in scope)+'\n')
    for table in ["main_table_rows", "paired_table_rows", "scope_table_rows"]:
        p=OUT/(table+".tex");p.write_text(p.read_text()+r"\bottomrule"+"\n")
    draw_figures(scope)
    (OUT/'cell_sources.json').write_text(json.dumps({'inputs':hashes,'cells':CELLS,'verified_raw_system_metric_means':checks,'verified_target_interactions':interaction_checks,'pending':['pt384_c96_gplus','pt384_l48_gplus'],'unrun':['pt96_c96_gplus']},indent=2)+'\n')
    print(f'Generated {len(CELLS)} numeric fields; checked {checks} raw-score system/metric means.')

def draw_figures(scope):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    def val(k):return CELLS[k]['value']
    # Figure 2: absolute means are descriptive, paired intervals are separate.
    fig,ax=plt.subplots(2,2,figsize=(6.6,3.35),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained')
    cols=['#245a81','#799cb5','#9b5d16','#d4af7b']
    for j,(s,title) in enumerate([('c96','Confirm96-B: primary interaction'),('l48','Length48: secondary interaction')]):
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
    fig,ax=plt.subplots(figsize=(6.6,4.3));fig.subplots_adjust(left=.52,right=.98,top=.98,bottom=.12)
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
