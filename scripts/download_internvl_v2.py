"""Fetch the explicitly requested ungated official fallback at its resolved SHA."""
import json,time
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from huggingface_hub import snapshot_download
from src.storage import atomic_json,digest
from src.session import update

meta=json.loads(Path('reports/evidence/v2/internvl_source_probe.json').read_text())
update('A official InternVL fallback download',revision=meta['revision'])
start=time.perf_counter()
snapshot=Path(snapshot_download(meta['model_id'],revision=meta['revision'],allow_patterns=['*.json','*.py','*.safetensors','*.model','*.txt','*.md','LICENSE'],max_workers=2))
record={**meta,'download_wall_seconds':time.perf_counter()-start,'source_checksums':{p.name:digest(p.read_bytes()) for p in snapshot.glob('*.py')},
        'weight_checksums':{p.name:digest(p.read_bytes()) for p in snapshot.glob('*.safetensors')},'snapshot_path_local_only':str(snapshot)}
Path('outputs/v2').mkdir(parents=True,exist_ok=True)
atomic_json('outputs/v2/internvl_download.json',record)
record.pop('snapshot_path_local_only')
atomic_json('reports/evidence/v2/internvl_download.json',record)
print(snapshot,flush=True)
