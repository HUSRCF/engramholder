"""Read-only selection audit; no predicted structures are used."""
import json,pathlib,hashlib,collections,datetime
from engramfold.evaluation.blast_filter import parse_hsp
r=pathlib.Path('/media/PM982/engramfold/runs/protenix_fresh192_20260925');p=r/'panel';sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
lock=json.loads((p/'selection_lock.json').read_text());assert all(sha(f)==h for f,h in lock['hashes'].items())
ref=json.loads((p/'reference_manifest.json').read_text());inf=json.loads((p/'inference_manifest.json').read_text());t=ref['targets'];ids={x['target_id'] for x in t};assert len(t)==len(ids)==192
assert collections.Counter(x['length_bin']for x in t)=={'128-191':48,'192-255':48,'256-319':48,'320-384':48}
assert len({x['pdb_id']for x in t})==192 and len({x['sequence']for x in t})==192
ex=json.loads((p/'exclusions.json').read_text());oldids=set(ex['target_ids']);qseq=set(ex['query_sequences']);pdbs={i[:4].lower()for i in oldids};assert not(ids&oldids) and not({x['pdb_id'].lower()for x in t}&pdbs) and not({x['sequence']for x in t}&qseq)
for a,b in zip(t,inf['targets']):
 assert set(b)=={'target_id','sequence','sequence_sha256','sequence_length','length_bin'}
 assert all(a[k]==v for k,v in b.items());assert len(a['sequence'])==a['sequence_length'];assert hashlib.sha256(a['sequence'].encode()).hexdigest()==a['sequence_sha256']
 assert a['initial_release_date']<='2021-09-30' and a['reference_ca_coverage']>=.9 and a['mapping_status']=='verified';assert sha(a['raw_mmcif_path'])==a['raw_mmcif_sha256']
base={x['id']:x for x in ex['references']};records=json.loads((p/'search_records.json').read_text());targetlookup={x['target_id']:x for bucket in json.loads((p/'ordered_candidates.json').read_text())for x in bucket}
for x in records:
 cmd=x['command'];d={cmd[i]:cmd[i+1] for i in range(1,len(cmd),2)};assert d['-evalue']=='0.001' and d['-comp_based_stats']=='2' and '-max_hsps' not in d
 folder=pathlib.Path(d['-db']).parent
 extra=json.loads((folder/'panel_entries.json').read_text()) if x['label']=='panel' else {}
 mapping={**base,**extra};assert int(d['-max_target_seqs'])==len(mapping)
 assert sha(d['-out'])==x['hsps_sha256'] and sha(d['-query'])==x['query_sha256'];hits=[]
 for line in pathlib.Path(d['-out']).read_text().splitlines():
  h=parse_hsp(line);refrow=mapping[h['sseqid']];candidate=targetlookup[x['target_id']]['sequence'];assert h['qlen']==len(candidate) and h['slen']==len(refrow['sequence'])
  assert h['qseq'].replace('-','').upper()==candidate[h['qstart']-1:h['qend']];assert h['sseq'].replace('-','').upper()==refrow['sequence'][h['sstart']-1:h['send']]
  denom=min(h['qlen'],h['slen']) if refrow['known_query'] else h['qlen'];qualifies=h['evalue']<=1e-5 and h['paired']>=50 and h['nident']/h['length']>=.3 and h['paired']/denom>=.7
  if qualifies and (x['label']=='exposure' or h['sseqid']in extra):hits.append(h)
 assert hits==x['hits']
 if x['target_id']in ids:assert not hits
for folder in p.glob('db_after_*'):
 hashes=json.loads((folder/'hashes.json').read_text());assert all(sha(k)==v for k,v in hashes.items())
dec=json.loads((p/'decisions.json').read_text());assert [x['target_id']for x in dec if x['decision']=='accepted']==[x['target_id']for x in t]
for binname in ['128-191','192-255','256-319','320-384']:
 ts=[x for x in t if x['length_bin']==binname];assert [x['rank']for x in ts]==sorted(x['rank']for x in ts)
 for x in ts:assert x['rank']==hashlib.sha256(('protenix-fourcell-fresh192-v1|'+x['target_id']).encode()).hexdigest()
result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_lock_sha256=sha(p/'selection_lock.json'),n_targets=192,quota=[48]*4,length_range=[min(x['sequence_length']for x in t),max(x['sequence_length']for x in t)],known_id_pdb_sequence_overlap=0,independently_rechecked_searches=len(records),all_recorded_hsp_and_database_hashes_pass=True,no_predicted_scores_used=True,family_isolation=False)
(r/'selection_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
