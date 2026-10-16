"""Prepare or validate the fixed development partition without model outputs."""
import json
from pathlib import Path
from scripts.prepare_debug import prepare_ref,prepare_gqa
from src.data import validate_manifest
from src.storage import atomic_json

rows=[]
for task,fn in (("refcocog",prepare_ref),("gqa",prepare_gqa)):
    p=Path(f"outputs/manifests/{task}_development.json")
    items=json.loads(p.read_text()) if p.exists() else fn("development",40,20262)
    if len(items)!=40: raise RuntimeError("Incomplete development checkpoint; inspect before resampling")
    rows+=items
all_rows=rows+json.loads(Path("outputs/manifests/gqa_debug.json").read_text())+json.loads(Path("outputs/manifests/refcocog_debug.json").read_text())
result=validate_manifest(all_rows,Path("datasets"))
atomic_json("reports/evidence/development_manifest_summary.json",{**result,"development_count":80,
            "status":"VALIDATED_NO_OUTCOME_BASED_SAMPLING","items":[{k:r[k] for k in ("dataset","image_id","item_id","image_sha256")} for r in rows]})
print(result)
