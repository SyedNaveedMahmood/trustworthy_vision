"""Local structural/perimeter diagnostics, never semantic edit labels."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,sobel
from skimage.metrics import structural_similarity
from src.edits.pilot import masks
from src.storage import atomic_json
from src.session import update


def run():
    edits=json.loads(Path('outputs/v2/edits_pairs_corrected.json').read_text());manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items']
    out=Path('reports/evidence/v2/edit_structural_statistics.json');rows=json.loads(out.read_text())['rows'] if out.exists() else []
    done={(r['key'],r['condition']) for r in rows}
    for i,e in enumerate(edits):
        item=next(x for x in manifest if x['dataset']==e['task'] and x['image_id']==e['image_id']);image=Image.open(item['canonical_path']).convert('RGB');original=np.asarray(image)
        core,mask,_=masks(image.size,e['box'],e['params']['halo']);core=np.asarray(core)>0;mask=np.asarray(mask)>0;boundary=mask&~binary_erosion(mask,iterations=2)
        gray=original.mean(axis=2);gradient=np.hypot(sobel(gray,0),sobel(gray,1))
        for condition in ['removal','sham']:
            if (e['key'],condition) in done:continue
            changed=np.asarray(Image.open(e[condition+'_path']).convert('RGB'))
            ssim,ssim_map=structural_similarity(original,changed,channel_axis=2,data_range=255,full=True)
            delta=abs(changed.astype(float)-original)
            rows.append({'task':e['task'],'image_id':e['image_id'],'operator':e['operator'],'region_index':e['region_index'],'key':e['key'],'condition':condition,
                         'ssim_whole_image':float(ssim),'ssim_core_mean':float(ssim_map[core].mean()),'ssim_expanded_mean':float(ssim_map[mask].mean()),
                         'boundary_rgb_absolute_change_mean':float(delta[boundary].mean()),'original_core_gradient_mean':float(gradient[core].mean()),
                         'original_control_or_candidate_mask_area':int(core.sum()),'semantic_validity':'HUMAN_AUDIT_PENDING'})
            done.add((e['key'],condition))
        if i%12==0:
            atomic_json(out,{'status':'RUNNING','rows':rows});update('D structural edit diagnostics',completed=len(rows))
    summary={}
    for op in ['occlusion','blur','lama']:
        for condition in ['removal','sham']:
            subset=[r for r in rows if r['operator']==op and r['condition']==condition]
            summary[f'{op}/{condition}']={key:float(np.mean([r[key] for r in subset])) for key in ['ssim_whole_image','ssim_core_mean','ssim_expanded_mean','boundary_rgb_absolute_change_mean']}
    atomic_json(out,{'status':'COMPLETE_COMPUTATIONAL_ONLY','metric':'scikit-image0.25.2 structural_similarity default window7, data_range255, channel_axis2','rows':rows,'summary':summary,
                    'limitation':'SSIM maps use local windows crossing the mask boundary. Whole-image high SSIM is dominated by unchanged pixels and never proves semantic validity.'})
    update('D structural edit diagnostics',status='COMPLETE_COMPUTATIONAL_ONLY',completed=len(rows))


if __name__=='__main__':run()
