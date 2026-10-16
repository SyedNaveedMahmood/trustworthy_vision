"""Resume range downloads from the official PyTorch host and verify its SHA-256.

This repairs stalled streamed uv downloads; no mirror or unverifiable binary.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

from src.storage import atomic_json

URL="https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp311-cp311-win_amd64.whl"
SIZE=3273066072
SHA="138c66dcd0ed2f07aafba3ed8b7958e2bed893694990e0b4b55b6b2b4a336aa6"
CHUNK=64*1024*1024
ROOT=Path("outputs/packages"); ROOT.mkdir(parents=True,exist_ok=True)


def chunk(index):
    start=index*CHUNK; end=min(SIZE,start+CHUNK)-1; length=end-start+1
    path=ROOT/f"torch.part{index:03}"
    if path.exists() and path.stat().st_size==length: return path
    for attempt in range(3):
        # A partial individual range is overwritten, complete ranges are retained.
        p=subprocess.run(["curl.exe","-fLsS","--connect-timeout","15","--max-time","240",
                          "--range",f"{start}-{end}",URL,"--output",str(path)],capture_output=True,text=True)
        if p.returncode==0 and path.stat().st_size==length: return path
        if attempt<2: time.sleep(2**attempt)
    raise RuntimeError(f"Range {index} failed after three attempts: {p.stderr[:200]}")


if __name__=="__main__":
    count=(SIZE+CHUNK-1)//CHUNK
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(chunk,i) for i in range(count)]
        for done,f in enumerate(as_completed(futures),1):
            f.result(); print(f"Official CUDA wheel ranges {done}/{count}",flush=True)
    output=ROOT/"torch-2.7.1+cu128-cp311-cp311-win_amd64.whl"
    h=hashlib.sha256()
    with output.open("wb") as target:
        for i in range(count):
            with (ROOT/f"torch.part{i:03}").open("rb") as source:
                while block:=source.read(1024*1024):
                    h.update(block); target.write(block)
    if h.hexdigest()!=SHA: raise RuntimeError("Official wheel checksum mismatch")
    atomic_json("reports/evidence/cuda_wheel_download.json",{"url":URL,"bytes":SIZE,"sha256":SHA,
                "verified":True,"wall_seconds":time.perf_counter()-start,"range_count":count})
    print("Verified official CUDA wheel SHA-256",flush=True)
