"""Redact machine-local paths in published failure records; raw logs stay local."""
import json
from pathlib import Path
from src.storage import atomic_json

workspace=str(Path.cwd())
home=str(Path.home())


def scrub(value):
    if isinstance(value,str):
        for source,target in ((workspace,"<workspace>"),(home,"<user-home>")):
            value=value.replace(source,target).replace(source.replace(chr(92),"/"),target)
        return value
    if isinstance(value,list): return [scrub(v) for v in value]
    if isinstance(value,dict): return {k:scrub(v) for k,v in value.items()}
    return value


for p in Path("reports/evidence").glob("*.json"):
    atomic_json(p,scrub(json.loads(p.read_text(encoding="utf-8"))))
for p in (Path("configs/requirements-vlm.lock.txt"),Path("reports/evidence/vlm_environment.txt")):
    lines=p.read_text(encoding="utf-8-sig").splitlines()
    p.write_text("\n".join("torch==2.7.1+cu128" if l.startswith("torch @") else l for l in lines)+"\n",encoding="utf-8")
