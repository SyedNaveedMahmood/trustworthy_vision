"""Budget-bounded corrected-Smol dose study, fixed first five images/task."""
import json
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.edits.pilot import request
from scripts.run_ablations_v2 import scaled_box
from src.storage import atomic_json,digest
from src.session import update,check_window


def run():
    manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items'];items=[r for t in ['gqa','refcocog'] for r in [x for x in manifest if x['dataset']==t][:5]]
    source=json.loads(Path('reports/evidence/v2/E2_pairs_corrected_smol.json').read_text());clean={(r['task'],r['image_id']):r for r in source['rows'] if r['variant']=='clean'}
    path=Path('reports/evidence/v2/E4_smol_bounded.json');record=json.loads(path.read_text()) if path.exists() else {'rows':[],'status':'RUNNING','manifest_sha256':digest(items),'selection':'fixed_first5_initial_images_per_task; no outcome filtering','maximum_new_forwards':160,'human_audit':'HUMAN_AUDIT_PENDING'}
    if record['manifest_sha256']!=digest(items):raise ValueError('Bounded E4 manifest changed')
    done={(r['task'],r['image_id'],r['factor'],r['variant']) for r in record['rows']};budget=json.loads(Path('outputs/budget.json').read_text())
    remaining=160-sum(r['variant']!='clean' for r in record['rows'])
    if budget['forward_calls']+remaining>12000:raise RuntimeError('Insufficient reserved forward budget for complete bounded Smol study')
    tm=Telemetry()
    with tm:
        p=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models']['smol'];model=ChoiceModel(p['model_id'],p['revision'],'smol',long_edge=768)
        for item in items:
            task=item['dataset'];iid=item['image_id'];image=Image.open(item['canonical_path']).convert('RGB');reference=clean[task,iid]
            rendered,_,_,_=present(image,item)
            if digest(rendered.tobytes()+str(rendered.size).encode())!=reference['image_sha256']:raise ValueError('Clean reuse input mismatch')
            for factor,op,params,scale in [('blur12','blur',{'sigma':12},1),('blur36','blur',{'sigma':36},1),('mask075','occlusion',{},.75),('mask125','occlusion',{},1.25)]:
                cell=(task,iid,factor,'clean')
                if cell not in done:
                    record['rows'].append({**reference,'factor':factor,'variant':'clean','reused_from_complete_E2':True});done.add(cell)
                for ri,box in enumerate(item['regions'][:2]):
                    edit=request(image,scaled_box(box,image.size,scale),op,**params)
                    for condition in ['removal','sham']:
                        check_window();variant=f'r{ri}_{condition}';cell=(task,iid,factor,variant)
                        if cell in done:continue
                        im,q,options,gold=present(Image.open(edit[condition+'_path']).convert('RGB'),item);row=model.score(im,q,options)
                        row.update(task=task,image_id=iid,factor=factor,variant=variant,gold=gold,correct=row['prediction']==gold,operator=op,mask_linear_scale=scale)
                        record['rows'].append(row);done.add(cell);atomic_json(path,record)
            update('F corrected Smol bounded E4',completed=len(done))
    record.update(status='COMPLETE_EXPLORATORY',telemetry=tm.result,cumulative_budget=model.budget.value);atomic_json(path,record)


if __name__=='__main__':run()
