"""Prepare image-disjoint qualification inputs, no inference/outcome access."""
import json
from pathlib import Path
from scripts.prepare_debug import prepare_ref,prepare_gqa
from src.data import validate_manifest
from src.storage import atomic_json
from src.session import update

update('A independent qualification data')
all_rows=[]
for task,fn in [('refcocog',prepare_ref),('gqa',prepare_gqa)]:
    p=Path(f'outputs/manifests/{task}_qualification.json')
    rows=json.loads(p.read_text()) if p.exists() and len(json.loads(p.read_text()))==20 else fn('qualification',20,20265)
    all_rows+=rows
    atomic_json(f'reports/evidence/v2/{task}_qualification_manifest.json',{'count':len(rows),'items':[{k:r[k] for k in ['dataset','image_id','image_sha256','item_id','seed']} for r in rows]})
for p in Path('outputs/manifests').glob('*_debug.json'): all_rows+=json.loads(p.read_text())
for p in Path('outputs/manifests').glob('*_development.json'): all_rows+=json.loads(p.read_text())
result=validate_manifest(all_rows,Path('datasets'))
atomic_json('reports/evidence/v2/qualification_split_check.json',result)
update('A independent qualification data',status='COMPLETE',**result)
