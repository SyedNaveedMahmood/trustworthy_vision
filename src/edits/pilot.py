"""Content-hashed provisional removal/sham pipeline and label-free controls."""
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from ..storage import digest,atomic_json
from . import box_mask

LAMA_SHA='786f5936b27fb3dacd2b1ad799e4de968ea697e7'


def area(box): return max(0,box[2]-box[0])*max(0,box[3]-box[1])


def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1,area(a)+area(b)-inter)


def independent_control(size,candidate,seed):
    """Only geometry/seed inputs. Does not know annotations, question, gold or outputs."""
    w=math.ceil(candidate[2])-math.floor(candidate[0]); h=math.ceil(candidate[3])-math.floor(candidate[1])
    if w>=size[0] or h>=size[1]: raise ValueError('No equal-area translated control fits')
    rng=np.random.default_rng(int(digest([list(size),list(candidate),seed])[:16],16))
    for _ in range(256):
        x=int(rng.integers(0,size[0]-w+1)); y=int(rng.integers(0,size[1]-h+1))
        control=(x,y,x+w,y+h)
        if overlap(candidate,control)==0: return control
    raise ValueError('No nonoverlapping independent control in 256 fixed proposals')


def grid_candidates(size):
    w,h=size; bw,bh=round(w*.25),round(h*.25)
    return [(round(w*cx)-bw//2,round(h*cy)-bh//2,round(w*cx)-bw//2+bw,round(h*cy)-bh//2+bh)
            for cy,cx in [(0.25,0.25),(.25,.75),(.75,.25),(.75,.75)]]


def masks(size,box,halo=8):
    core=box_mask(size,box)
    expanded=(max(0,math.floor(box[0])-halo),max(0,math.floor(box[1])-halo),min(size[0],math.ceil(box[2])+halo),min(size[1],math.ceil(box[3])+halo))
    return core,box_mask(size,expanded),expanded


def request(image,box,operator,*,sigma=24,fill='mean',halo=8):
    image=image.convert('RGB'); core,mask,expanded=masks(image.size,box,halo)
    params={'operator':operator,'sigma':sigma,'fill':fill,'halo':halo,'revision':LAMA_SHA if operator=='lama' else 'pil_halo_restore_v2'}
    key=digest({'image':digest(image.tobytes()+str(image.size).encode()),'core':digest(core.tobytes()),'mask':digest(mask.tobytes()),'params':params})
    root=Path('outputs/v2/edits')/key
    meta={'key':key,'box':list(box),'expanded_box':list(expanded),'params':params,'core_mask_sha256':digest(core.tobytes()),'mask_sha256':digest(mask.tobytes()),
          'removal_path':str(root/'remove.png').replace(chr(92),'/'),'sham_path':str(root/'sham.png').replace(chr(92),'/')}
    if (root/'metadata.json').exists():
        import json
        saved=json.loads((root/'metadata.json').read_text())
        if all(Path(saved[c+'_path']).exists() and digest(Path(saved[c+'_path']).read_bytes())==saved[c+'_sha256'] for c in ['removal','sham']): return saved
        raise ValueError('Edited-image checkpoint corrupted; inspect instead of silently replacing')
    root.mkdir(parents=True,exist_ok=True)
    if operator=='lama':
        inp=Path('outputs/v2/lama/input'); inp.mkdir(parents=True,exist_ok=True)
        image.save(inp/f'{key}.png'); mask.save(inp/f'{key}_mask001.png')
        raw=Path('outputs/v2/lama/output')/f'{key}_mask001.png'
        if not raw.exists(): return meta
        modified=Image.open(raw).convert('RGB')
        if modified.size!=image.size: raise ValueError('Official LaMa unpadding misalignment')
    elif operator=='occlusion':
        mean=tuple(int(x) for x in np.asarray(image).reshape(-1,3).mean(axis=0).round()) if fill=='mean' else (128,128,128) if fill=='gray128' else (127,127,127)
        modified=Image.new('RGB',image.size,mean)
    elif operator=='blur': modified=image.filter(ImageFilter.GaussianBlur(sigma))
    else: raise ValueError(operator)
    original=np.asarray(image).astype(int); rawdiff=np.abs(np.asarray(modified).astype(int)-original)
    removal=Image.composite(modified,image,mask)
    sham=Image.composite(image,removal,core)
    removal.save(root/'remove.png'); sham.save(root/'sham.png')
    core_arr=np.asarray(core)>0; mask_arr=np.asarray(mask)>0
    remdiff=np.abs(np.asarray(removal).astype(int)-original); shamdiff=np.abs(np.asarray(sham).astype(int)-original)
    meta.update(removal_sha256=digest((root/'remove.png').read_bytes()),sham_sha256=digest((root/'sham.png').read_bytes()),
                core_fraction=float(core_arr.mean()),mask_fraction=float(mask_arr.mean()),
                mask_perimeter_pixels=int(np.count_nonzero(np.diff(mask_arr.astype(int),axis=0))+np.count_nonzero(np.diff(mask_arr.astype(int),axis=1))),
                outside_mask_max_change=int(remdiff[~mask_arr].max()) if (~mask_arr).any() else 0,
                sham_core_max_change=int(shamdiff[core_arr].max()),
                raw_lama_outside_max_change=int(rawdiff[~mask_arr].max()) if operator=='lama' and (~mask_arr).any() else None,
                mean_abs_change_inside=float(remdiff[mask_arr].mean()),
                removal_identical=bool(not remdiff.any()),sham_identical=bool(not shamdiff.any()),semantic_validity='HUMAN_AUDIT_PENDING')
    atomic_json(root/'metadata.json',meta)
    return meta
