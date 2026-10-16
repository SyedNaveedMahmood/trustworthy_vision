"""Independent official CPU inference replay on a fixed real pilot edit."""
import json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
from src.storage import atomic_json,digest
from src.session import update


def run():
    edits=json.loads(Path('outputs/v2/edits_pairs_corrected.json').read_text());e=next(x for x in edits if x['operator']=='lama' and x['region_index']==0)
    root=Path.cwd();repo=root/'third_party/repos/lama';indir=root/'outputs/v2/lama/replay/input';outdir=root/'outputs/v2/lama/replay/output'
    indir.mkdir(parents=True,exist_ok=True);outdir.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','_mask001.png']:shutil.copy2(root/'outputs/v2/lama/input'/f"{e['key']}{suffix}",indir/f"{e['key']}{suffix}")
    env=os.environ.copy();env.update(PYTHONPATH=str(repo),TORCH_HOME=str(root/'third_party_weights'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[str(root/'.venv-lama/Scripts/python.exe'),str(repo/'bin/predict.py'),f"model.path={(root/'third_party_weights/big-lama').as_posix()}",f'indir={indir.as_posix()}',f'outdir={outdir.as_posix()}']
    start=time.perf_counter()
    with (root/'outputs/v2/lama/replay/official.log').open('w',encoding='utf-8') as stream:r=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Official real-image replay failed')
    original=root/'outputs/v2/lama/output'/f"{e['key']}_mask001.png";replay=outdir/f"{e['key']}_mask001.png"
    a=np.asarray(Image.open(original));b=np.asarray(Image.open(replay))
    result={'status':'REAL_IMAGE_REPLAY','task':e['task'],'image_id':e['image_id'],'edit_key':e['key'],'device':'cpu','source_revision':e['params']['revision'],
            'official_subprocess_wall_seconds':time.perf_counter()-start,'original_sha256':digest(original.read_bytes()),'replay_sha256':digest(replay.read_bytes()),
            'byte_identical':original.read_bytes()==replay.read_bytes(),'pixel_max_absolute_error':int(abs(a.astype(int)-b.astype(int)).max()),'shape':list(a.shape),
            'semantic_validity':'HUMAN_AUDIT_PENDING'}
    atomic_json('reports/evidence/v2/lama_real_replay.json',result);update('D official real-image replay',status='COMPLETE',byte_identical=result['byte_identical'])


if __name__=='__main__':run()
