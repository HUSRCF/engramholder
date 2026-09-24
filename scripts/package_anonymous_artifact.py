#!/usr/bin/env python3
"""Create a bounded source/score artifact without personal paths or Git history."""
import argparse,hashlib,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--name',default='anonymous_artifact');args=parser.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+',args.name)
DEST=ROOT/'build'/args.name
if DEST.exists(): raise FileExistsError('Preserve existing archive; choose a new output directory/version')
DEST.mkdir(parents=True)

def redact(s):
    s=re.sub(r'https?://github\.com/HUSRCF/[^\s"<>]+','[personal repository omitted]',s,flags=re.I)
    s=re.sub(r'/home/(?:husrcf|pc)(?=/|\b)','/workspace',s)
    s=re.sub(r'/data/user/shuang886(?=/|\b)','/workspace',s)
    s=s.replace('DiamondHill','ROCm host').replace('Precision','secondary ROCm host')
    return s

BINARY_SUFFIXES={'.pdf','.png','.pt','.npz','.gz'}

def write(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src.suffix in BINARY_SUFFIXES:dst.write_bytes(src.read_bytes())
    else:dst.write_text(redact(src.read_text()))

for name in ['iclr2027_conference.tex','iclr2027_conference.sty','iclr2027_conference.bst','math_commands.tex','natbib.sty','fancyhdr.sty','paper_references.bib']:
    write(ROOT/name,DEST/name)
for directory in ['sections','appendices','figures','generated','reproducibility']:
    for p in (ROOT/directory).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:write(p,DEST/p.relative_to(ROOT))
lockpath=Path('notes/writing_branch_20260922/paper_sources.v8.lock.json')
inputs=json.loads((ROOT/lockpath).read_text())
for name in inputs:write(ROOT/name,DEST/name)
# Only non-scientific identifying strings are redacted; revalidate numerics below.
lock={name:hashlib.sha256((DEST/name).read_bytes()).hexdigest() for name in inputs}
(DEST/lockpath).parent.mkdir(parents=True,exist_ok=True)
(DEST/lockpath).write_text(json.dumps(lock,indent=2)+'\n')
(DEST/lockpath.parent/'bundled_source_provenance.json').write_text(json.dumps({name:dict(original_sha256=inputs[name],bundled_sha256=lock[name]) for name in inputs},indent=2)+'\n')
write(ROOT/'scripts/build_paper_assets.py',DEST/'scripts/build_paper_assets.py')
write(ROOT/'scripts/verify_openfold_esmc_A.py',DEST/'scripts/verify_openfold_esmc_A.py')
for name in ['verify_diamondhill_fourcells.py','diamondhill_paper_assets.py','paper_figure_layouts.py','verify_e1_prediction.py']:
    write(ROOT/'scripts'/name,DEST/'scripts'/name)
write(ROOT/'scripts/analyze_openfold_followups.py',DEST/'scripts/analyze_openfold_followups.py')
write(ROOT/'tests/test_numeric_keys.py',DEST/'tests/test_numeric_keys.py')
write(ROOT/'tests/test_openfold_followups.py',DEST/'tests/test_openfold_followups.py')
write(ROOT/'tests/test_single_prediction_score.py',DEST/'tests/test_single_prediction_score.py')
write(ROOT/'tests/test_e1_prediction.py',DEST/'tests/test_e1_prediction.py')
(DEST/'README.md').write_text((DEST/'reproducibility/README.md').read_text())
source_manifest=json.loads((DEST/'reproducibility/source_manifest.json').read_text())
for name,row in source_manifest.items():
    row['original_snapshot_sha256']=row.pop('sha256')
    row['sha256']=hashlib.sha256((DEST/'reproducibility'/name).read_bytes()).hexdigest()
    row['redaction']='Only host/user/path strings, when present; algorithms unchanged.'
(DEST/'reproducibility/source_manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
subprocess.run([sys.executable,str(DEST/'scripts/build_paper_assets.py')],check=True,cwd=DEST)
subprocess.run([sys.executable,str(DEST/'tests/test_numeric_keys.py')],check=True,cwd=DEST)
subprocess.run([sys.executable,str(DEST/'tests/test_openfold_followups.py')],check=True,cwd=DEST)
subprocess.run([sys.executable,str(DEST/'tests/test_single_prediction_score.py')],check=True,cwd=DEST)
subprocess.run([sys.executable,str(DEST/'tests/test_e1_prediction.py')],check=True,cwd=DEST)
subprocess.run([sys.executable,str(DEST/'reproducibility/operator_smoke.py')],check=True,cwd=DEST)
a=json.loads((ROOT/'generated/cell_sources.json').read_text())['cells']
b=json.loads((DEST/'generated/cell_sources.json').read_text())['cells']
assert {k:v['value'] for k,v in a.items()}=={k:v['value'] for k,v in b.items()}
for p in DEST.rglob('*'):
    if p.is_file() and p.suffix not in BINARY_SUFFIXES|{'.pyc'}:
        assert not re.search(r'HUSRCF|husrcf|shuang886|/home/pc\b|BEGIN [A-Z ]*PRIVATE KEY',p.read_text()),p
# No bytecode, Git metadata, personal README, old prose, raw cluster logs, or weights.
files=[p for p in DEST.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(DEST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=ROOT/'build'/(args.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in [*files,DEST/'MANIFEST.json']:z.write(p,Path('artifact')/p.relative_to(DEST))
print(json.dumps({'archive':str(archive),'files':len(files)+1,'numerical_cells_unchanged':len(a),'scope':'score/figure reconstruction and operator check; not full retraining'},indent=2))
