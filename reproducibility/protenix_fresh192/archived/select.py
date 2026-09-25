"""Fresh96 selection v2; reused calibrated BLAST contract, no folding scores."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from engramfold.data.diagnostic_manifest import DiagnosticSelection,_eligible,_iter_jsonl
from engramfold.data.independent_panel import reference_mapping
from engramfold.evaluation.blast_filter import BLAST_FIELDS,parse_hsp,qualifying_hsp
def inference_target(t):
 return {k:t[k] for k in ("target_id","sequence","sequence_sha256","sequence_length","length_bin")}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def fasta(p,rows):p.write_text(''.join('>'+r['id']+'\n'+r['sequence']+'\n' for r in rows))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();out=a.root/'panel';out.mkdir(parents=True,exist_ok=False);(out/'searches').mkdir();start=time.monotonic()
 runs=Path('/media/PM982/engramfold/runs');cal=runs/'length_filter_calibration_20260920';tools=Path('/media/PM982/engramfold/tools/blast_2_17/ncbi-blast-2.17.0+/bin');cat=Path('/media/PM982/onestepfold/catalog_v1/monomer_candidates.jsonl.gz');raw=Path('/media/PM982/onestepfold/Dataset/raw/pdb_mmcif')
 data=json.loads((cal/'inputs.json').read_text());assert json.loads((cal/'analysis.json').read_text())['accepted'];cal_lock=json.loads((cal/'execution_lock.json').read_text());assert sha(tools/'blastp')==cal_lock['hashes'][str(tools/'blastp')]
 exposure=a.root/'scoring_exposure_audit.json';audit=json.loads(exposure.read_text());assert audit['all_scans_parsed']
 ids=set(audit['target_ids']);query=set(audit['query_sequences']);pdbs={tid[:4].lower() for tid in ids};catalog=list(_iter_jsonl(cat))
 for t in catalog:
  tid=t['pdb_id'].lower()+'_'+t['source_label_asym_id']
  if tid in ids:query.add(t['sequence'])
 original_refs={x['sequence'] for x in data['references']};allrefs=sorted(original_refs|query)
 refs=[dict(id=f'ref{i:05d}',sequence=seq,known_query=seq in query) for i,seq in enumerate(allrefs)];refmap={x['id']:x for x in refs};seqs=set(allrefs)
 bins=[(128,191),(192,255),(256,319),(320,384)];buckets=[[]for _ in bins]
 for t in catalog:
  if not _eligible(t,DiagnosticSelection()) or t['initial_release_date']>'2021-09-30' or t['pdb_id'].lower()in pdbs or t['sequence']in seqs:continue
  t=dict(t);t['target_id']=t['pdb_id'].lower()+'_'+t['source_label_asym_id'];t['rank']=hashlib.sha256(('protenix-fourcell-fresh192-v1|'+t['target_id']).encode()).hexdigest()
  for k,(lo,hi) in enumerate(bins):
   if lo<=len(t['sequence'])<=hi:buckets[k].append(t)
 buckets=[sorted(b,key=lambda t:t['rank'])[:2048] for b in buckets];write(out/'ordered_candidates.json',buckets)
 fasta(out/'base_references.fasta',refs);(out/'base_db').mkdir()
 with (out/'base_db/makeblastdb.log').open('w') as f:subprocess.run([str(tools/'makeblastdb'),'-in',str(out/'base_references.fasta'),'-dbtype','prot','-parse_seqids','-out',str(out/'base_db/references')],check=True,stdout=f,stderr=subprocess.STDOUT)
 paths=[cat,exposure,cal/'inputs.json',cal/'execution_lock.json',cal/'analysis.json',a.root/'protocol.md',a.root/'model_lock.json',Path(__file__),a.source/'src/engramfold/evaluation/blast_filter.py',a.source/'src/engramfold/data/independent_panel.py',tools/'blastp',tools/'makeblastdb',out/'ordered_candidates.json',out/'base_references.fasta',*sorted((out/'base_db').glob('*'))]
 lock=dict(schema='engramfold.protenix_fresh192.selection.v2',utc=datetime.now(timezone.utc).isoformat(),hashes={str(f):sha(f)for f in paths},known_queries=len(query),references=len(refs),max_candidates_per_bin=2048,seconds_cap=10800,rank='sha256(protenix-fourcell-fresh192-v1|target_id)',family_isolation=False,reason='New approved P192: fixed Fresh96 BLAST v2 algorithm with refreshed exposure, 48 per stratum, new deterministic salt; no outcome-based selection.')
 write(out/'selection_lock.json',lock);write(out/'exclusions.json',dict(target_ids=sorted(ids),query_sequences=sorted(query),references=refs))
 selected=[];decisions=[];seen=set();aug=None;augmap={};search_records=[]
 def search(t,db,extra,label):
  folder=out/'searches'/f'{len(decisions):04d}_{label}';folder.mkdir();q=folder/'query.fasta';fasta(q,[dict(id='candidate',sequence=t['sequence'])]);result=folder/'hsps.tsv';mapping={**refmap,**extra}
  cmd=[str(tools/'blastp'),'-task','blastp','-query',str(q),'-db',str(db),'-out',str(result),'-outfmt','6 '+' '.join(BLAST_FIELDS),'-matrix','BLOSUM62','-gapopen','11','-gapextend','1','-word_size','3','-seg','yes','-soft_masking','true','-comp_based_stats','2','-evalue','0.001','-max_target_seqs',str(len(mapping)),'-num_threads','8']
  subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL);hits=[]
  for line in result.read_text().splitlines():
   r=parse_hsp(line);ref=mapping[r['sseqid']];assert r['qseqid']=='candidate' and r['qlen']==len(t['sequence']) and r['slen']==len(ref['sequence'])
   assert r['qseq'].replace('-','').upper()==t['sequence'][r['qstart']-1:r['qend']];assert r['sseq'].replace('-','').upper()==ref['sequence'][r['sstart']-1:r['send']]
   if qualifying_hsp(r,ref['known_query']) and (label=='exposure' or r['sseqid'] in extra):hits.append(r)
  search_records.append(dict(target_id=t['target_id'],label=label,command=cmd,query_sha256=sha(q),hsps_sha256=sha(result),hits=hits));return hits
 try:
  for bucket,(lo,hi) in zip(buckets,bins):
   count=0
   for t in bucket:
    if time.monotonic()-start>10800:raise TimeoutError('selection cap')
    reason=None;mapping=None;hits=[]
    if t['pdb_id'].lower() in seen:reason='duplicate_pdb'
    elif any(t['sequence']==s['sequence'] for s in selected):reason='duplicate_panel_sequence'
    else:
     hits=search(t,out/'base_db/references',{},'exposure')
     if hits:reason='exposure_sequence_hit'
     elif aug is not None:
      hits=search(t,aug,augmap,'panel')
      if hits:reason='panel_sequence_hit'
     if reason is None:
      try:mapping=reference_mapping(t,raw)
      except (ValueError,FileNotFoundError,RuntimeError) as e:reason='mapping:'+str(e)
    decisions.append(dict(target_id=t['target_id'],rank=t['rank'],decision=reason or 'accepted',matching_references=sorted({h['sseqid'] for h in hits})))
    if reason is None:
     selected.append({**t,**mapping,'sequence_length':len(t['sequence']),'sequence_sha256':hashlib.sha256(t['sequence'].encode()).hexdigest(),'length_bin':f'{lo}-{hi}'});seen.add(t['pdb_id'].lower());count+=1
     folder=out/f'db_after_{len(selected):02d}';folder.mkdir();augmap={f'ref{len(refs)+i:05d}':dict(id=f'ref{len(refs)+i:05d}',sequence=s['sequence'],known_query=True,target_id=s['target_id']) for i,s in enumerate(selected)};fasta(folder/'references.fasta',refs+list(augmap.values()));aug=folder/'references'
     with (folder/'makeblastdb.log').open('w') as f:subprocess.run([str(tools/'makeblastdb'),'-in',str(folder/'references.fasta'),'-dbtype','prot','-parse_seqids','-out',str(aug)],stdout=f,stderr=subprocess.STDOUT,check=True)
     write(folder/'hashes.json',{str(f):sha(f) for f in sorted(folder.glob('*'))});write(folder/'panel_entries.json',augmap)
    write(out/'decisions.json',decisions);write(out/'search_records.json',search_records);write(out/'progress.json',dict(accepted=len(selected),examined=len(decisions),length_bin=lo,seconds=time.monotonic()-start));print(json.dumps(dict(accepted=len(selected),examined=len(decisions),target=t['target_id'],decision=reason)),flush=True)
    if count==48:break
   if count!=48:raise RuntimeError(f'quota failed {lo}: {count}/48')
  assert len(selected)==192 and all(sha(f)==h for f,h in lock['hashes'].items())
  write(out/'reference_manifest.json',dict(schema='engramfold.protenix_fresh192.reference.v2',role='prospective_fixed_model_interaction_validation',target_count=192,selection_lock_sha256=sha(out/'selection_lock.json'),targets=selected,family_isolation=False,foundation_pretraining_overlap_unknown=True))
  write(out/'inference_manifest.json',dict(schema='engramfold.protenix_fresh192.sequence.v2',targets=[inference_target(t) for t in selected]))
  write(out/'complete.json',dict(passed=True,targets=192,reference_sha256=sha(out/'reference_manifest.json'),inference_sha256=sha(out/'inference_manifest.json'),selection_lock_sha256=sha(out/'selection_lock.json'),elapsed_seconds=time.monotonic()-start))
 except Exception as e:
  write(out/'stopped.json',dict(reason=repr(e),selected=len(selected),examined=len(decisions)));raise
if __name__=='__main__':main()
