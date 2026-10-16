"""Execute the unmodified official LaMa prediction script on an infrastructure fixture."""
from pathlib import Path
import os
import subprocess
import sys
import time

from PIL import Image
from src.edits import box_mask
from src.synthetic import shapes
from src.storage import atomic_json,digest

root=Path.cwd()
inputs=root/"outputs/lama_smoke/input"; outputs=root/"outputs/lama_smoke/output"
inputs.mkdir(parents=True,exist_ok=True); outputs.mkdir(parents=True,exist_ok=True)
image=shapes()[0]["image"].resize((192,128))
image.save(inputs/"fixture.png")
box_mask(image.size,(20,25,75,90)).save(inputs/"fixture_mask001.png")
env=os.environ.copy()
repo=root/"third_party/repos/lama"
env["PYTHONPATH"]=str(repo)
env["TORCH_HOME"]=str(root/"third_party_weights")
command=[str(root/".venv-lama/Scripts/python.exe"),str(repo/"bin/predict.py"),
         f"model.path={str(root/'third_party_weights/big-lama').replace(chr(92),'/')}",
         f"indir={str(inputs).replace(chr(92),'/')}",f"outdir={str(outputs).replace(chr(92),'/')}"]
start=time.perf_counter()
p=subprocess.run(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
Path("outputs/lama_smoke/official_predict.log").write_text(p.stdout,encoding="utf-8")
output=outputs/"fixture_mask001.png"
record={"source_commit":"786f5936b27fb3dacd2b1ad799e4de968ea697e7","returncode":p.returncode,
        "wall_seconds":time.perf_counter()-start,"device":"cpu","input_sha256":digest((inputs/"fixture.png").read_bytes()),
        "mask_sha256":digest((inputs/"fixture_mask001.png").read_bytes()),"semantic_validity":"NOT_AUDITED"}
if p.returncode==0 and output.exists():
    record.update(status="EXECUTED_NOT_SEMANTICALLY_VALIDATED",output_sha256=digest(output.read_bytes()))
else:
    record.update(status="FAILED",error_tail=p.stdout[-5000:])
atomic_json("reports/evidence/lama_smoke.json",record)
print(p.stdout[-5000:]); print(record["status"])
sys.exit(0 if record["status"].startswith("EXECUTED") else 2)
