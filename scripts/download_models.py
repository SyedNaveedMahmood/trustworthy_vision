"""Download pinned official primary weights without loading either model."""
import json
import os
from pathlib import Path
import time
import truststore
truststore.inject_into_ssl()
from huggingface_hub import snapshot_download
from src.storage import atomic_json

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]="60"
sources=json.loads(Path("reports/evidence/source_inventory.json").read_text(encoding="utf-8"))
for key,entry in sources["models"].items():
    record={"model_id":entry["model_id"],"revision":entry["revision"],"status":"DOWNLOADING","attempts":[]}
    out=Path(f"reports/evidence/{key}_download.json")
    atomic_json(out,record)
    start=time.perf_counter()
    for attempt in range(3):
        try:
            snapshot=snapshot_download(entry["model_id"],revision=entry["revision"],
                                       allow_patterns=["*.json","*.safetensors","*.model","*.txt","*.jinja"],max_workers=2)
            record.update(status="DOWNLOADED",file_count=sum(p.is_file() for p in Path(snapshot).rglob("*")),
                          total_bytes=sum(p.stat().st_size for p in Path(snapshot).rglob("*") if p.is_file()))
            break
        except Exception as e:
            record["attempts"].append({"attempt":attempt+1,"exception":type(e).__name__,"detail":str(e)})
            atomic_json(out,record)
            # Auth, inaccessible revision, certificate and deterministic errors need repair, not blind retry.
            if any(s in str(e).lower() for s in ("401", "403", "certificate", "revision not found")):
                record["status"]="BLOCKED"; break
            if attempt==2:
                record["status"]="FAILED"; break
            time.sleep(2**attempt)
    record["wall_seconds"]=time.perf_counter()-start
    atomic_json(out,record)
    print(key,record["status"],flush=True)
