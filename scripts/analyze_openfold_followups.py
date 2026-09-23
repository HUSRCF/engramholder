"""Reconstruct Fresh96 and the post-hoc training-setting interaction change.

Starts from locked score arrays, not structures. No experiment is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (20260923, 20260924, 20260925)
ROTATIONS = (20261001, 20261002, 20261003)
METRICS = ('ca_lddt', 'residue_ca_lddt', 'tm_score_fixed_full_length')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(records, panel, ids, metric, prefix=''):
    rows = [r for r in records if r['panel'] == panel]
    by = {(r['system'], r['target_id']): r for r in rows}
    assert len(by) == len(rows), 'Duplicate model-target record'
    def get(name):
        rs = [by[name, i] for i in ids]
        assert all(r['status'] == 'ok' for r in rs), 'Inspect failures explicitly'
        x = np.array([r[metric] for r in rs])
        assert np.isfinite(x).all()
        return x
    if prefix:
        f = np.stack([get(f'{prefix}_factor_native_s{s}') for s in SEEDS])
        g = np.stack([get(f'{prefix}_generic_plus_native_s{s}') for s in SEEDS])
        fr = np.stack([[get(f'{prefix}_factor_r{r}_s{s}') for r in ROTATIONS] for s in SEEDS])
        gr = np.stack([[get(f'{prefix}_generic_plus_r{r}_s{s}') for r in ROTATIONS] for s in SEEDS])
    else:
        f = np.stack([get(f'native_s{s}') for s in SEEDS])
        g = np.stack([get(f'gplus_s{s}') for s in SEEDS])
        fr = np.stack([[get(f'r{r}_s{s}') for r in ROTATIONS] for s in SEEDS])
        gr = np.stack([[get(f'gplus_r{r}_s{s}') for r in ROTATIONS] for s in SEEDS])
    df, dg = f[:, None] - fr, g[:, None] - gr
    return dict(F=f, RF=fr, G=g, RG=gr, df=df, dg=dg, psi=df-dg)


def summarize(cells, ids, draw):
    targets = cells.mean(axis=(0, 1))
    assert cells.shape == (3, 3, len(ids))
    return dict(mean=float(targets.mean()),
                ci95=np.quantile(targets[draw].mean(1), [.025, .975]).tolist(),
                target_ids=ids, per_target=targets.tolist(),
                positive_targets=int((targets > 0).sum()),
                per_seed=cells.mean((1, 2)).tolist(),
                per_rotation=cells.mean((0, 2)).tolist(),
                seed_rotation_cells=cells.mean(2).tolist(),
                positive_seed_rotation_cells=int((cells.mean(2) > 0).sum()))


def calculate(root):
    root = Path(root)
    load = lambda n: json.loads((root/'evidence'/f'{n}.json').read_text())
    fresh, fr = load('openfold_fresh96_summary'), load('openfold_fresh96_records')
    done = load('openfold_fresh96_complete')
    assert done['complete'] and done['analysis_sha256'] == sha(root/'evidence/openfold_fresh96_summary.json')
    assert done['records_sha256'] == sha(root/'evidence/openfold_fresh96_records.json')
    # Packaged redaction may change non-scientific path strings in a lock.
    lp = 'evidence/openfold_fresh96_execution_lock.json'
    actual = sha(root/lp)
    provenance = root/'notes/writing_branch_20260922/bundled_source_provenance.json'
    if provenance.exists():
        binding = json.loads(provenance.read_text())[lp]
        assert actual == binding['bundled_sha256']
        actual = binding['original_sha256']
    assert actual == fresh['execution_lock_sha256'] == done['execution_lock_sha256']
    ids = fresh['targets']
    assert len(ids) == len(set(ids)) == 96 and len(fr) == 2400
    assert fresh['failed'] == 0 and all(r['status'] == 'ok' for r in fr)
    assert len({(r['system'], r['target_id']) for r in fr}) == 2400
    assert {r['target_id'] for r in fr} == set(ids)
    models = load('openfold_fresh96_model_lock')['models']
    assert {r['system'] for r in fr} == {m['name'] for m in models} and len(models) == 25
    targets=load('openfold_fresh96_reference_manifest')['targets']
    assert [t['target_id'] for t in targets]==ids
    assert sorted([t['sequence_length'] for t in targets])[::95]==[129,384]
    bins={t['length_bin'] for t in targets}
    assert len(bins)==4 and all(sum(t['length_bin']==b for t in targets)==24 for b in bins)
    previous=[r for d in load('data_composition_audit')['datasets'].values() for r in d['records']]
    assert set(ids).isdisjoint({r['target_id'] for r in previous})
    assert {i.split('_')[0].lower() for i in ids}.isdisjoint({r['target_id'].split('_')[0].lower() for r in previous})
    assert {t['sequence_sha256'] for t in targets}.isdisjoint({r['sequence_sha256'] for r in previous})
    for t in targets:
        assert len(t['sequence'])==t['sequence_length']
        assert hashlib.sha256(t['sequence'].encode()).hexdigest()==t['sequence_sha256']
    draw = np.random.default_rng(20260921).integers(96, size=(20000, 96))
    fresh_results = {}
    for metric in METRICS:
        a = arrays(fr, 'fresh96', ids, metric)
        md = fresh['metrics'][metric]
        for key, arr in [('factor_rotation', a['df']), ('gplus_rotation', a['dg']), ('interaction', a['psi'])]:
            got = summarize(arr, ids, draw)
            for field in ('mean', 'ci95', 'per_target', 'positive_targets'):
                np.testing.assert_allclose(got[field], md[key][field], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['psi'].mean((1,2)), md['interaction_by_seed'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['psi'].mean((0,2)), md['interaction_by_rotation'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(a['psi'].mean(2), md['interaction_cells'], atol=1e-12, rtol=0)
        for key, source in [('F','factor_native'),('RF','factor_rotated'),('G','generic_plus_native'),('RG','generic_plus_rotated')]:
            assert abs(a[key].mean()-md['group_means'][source]) < 1e-12
        # Independently check every reported system mean and query contrast.
        for model, mean in md['system_means'].items():
            values = [r[metric] for r in fr if r['system'] == model]
            assert len(values) == 96 and abs(np.mean(values)-mean) < 1e-12
        q = np.array([next(r[metric] for r in fr if r['system']=='query' and r['target_id']==i) for i in ids])
        for key, values in [('native_minus_gplus',a['F']-a['G']),('factor_minus_query',a['F']-q),('generic_plus_minus_query',a['G']-q)]:
            v=values.mean(0)
            np.testing.assert_allclose(v,md[key]['per_target'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.quantile(v[draw].mean(1),[.025,.975]),md[key]['ci95'],atol=1e-12,rtol=0)
        fresh_results[metric] = summarize(a['psi'], ids, draw)
    old = load('openfold_gplus_rotation_records')
    new = load('openfold_esmc_A_records')
    protocol = json.loads((root/'evidence/openfold_training_psi_protocol.json').read_text())
    for p, digest in protocol['sources'].items(): assert sha(root/p) == digest
    training = {}
    for panel, n in [('confirm96',96),('length48',48)]:
        ids = sorted({r['target_id'] for r in old if r['panel']==panel})
        assert len(ids)==n and set(ids)=={r['target_id'] for r in new if r['panel']==panel}
        assert set(ids).isdisjoint(fresh['targets'])
        draw = np.random.default_rng(protocol['bootstrap']['seed']).integers(n,size=(protocol['bootstrap']['draws'],n))
        training[panel] = {}
        for metric in METRICS:
            a,b = arrays(old,panel,ids,metric), arrays(new,panel,ids,metric,'E_last')
            value = summarize(b['psi']-a['psi'],ids,draw)
            value['psi96_mean'], value['psi384_mean'] = float(a['psi'].mean()),float(b['psi'].mean())
            training[panel][metric] = value
    return dict(passed=True,fresh96=dict(predictions=2400,failures=0,metrics=fresh_results),
                training_setting_change=dict(exploratory=True,protocol='evidence/openfold_training_psi_protocol.json',panels=training),
                scope='Score-array verification and post-hoc paired target analysis; no CIF rescoring; no retraining uncertainty included.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=calculate(root)
    out=root/'evidence/openfold_followup_analysis.json'
    if args.check: assert result==json.loads(out.read_text())
    else: out.write_text(json.dumps(result,indent=2)+'\n')
    for panel,md in result['training_setting_change']['panels'].items():
        print(panel,md['ca_lddt']['mean'],md['ca_lddt']['ci95'])
    print('Fresh96 reconstruction passed; 2400 records, three metrics.')
