"""Bounded one-factor exploratory E4; first six geometry-selected images/task."""
import argparse,json
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.edits.pilot import request
from src.storage import atomic_json,digest
from src.session import update,check_window


def scaled_box(box,size,factor):
    x=(box[0]+box[2])/2;y=(box[1]+box[3])/2;w=(box[2]-box[0])*factor;h=(box[3]-box[1])*factor
    return [max(0,x-w/2),max(0,y-h/2),min(size[0],x+w/2),min(size[1],y+h/2)]


def run(key):
    manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items']
    items=[r for task in ['gqa','refcocog'] for r in [x for x in manifest if x['dataset']==task][:6]]
    edits=json.loads(Path('outputs/v2/edits_pairs_corrected.json').read_text())
    out=Path(f'reports/evidence/v2/E4_{key}.json');record=json.loads(out.read_text()) if out.exists() else {'rows':[],'status':'RUNNING','model':key,'selection':'first_six_per_task_before_E2_outcomes','manifest_sha256':digest(items),'human_audit':'HUMAN_AUDIT_PENDING'}
    if record['manifest_sha256']!=digest(items): raise ValueError('Ablation manifest changed')
    done={(r['task'],r['image_id'],r['factor'],r['variant']) for r in record['rows']};tm=Telemetry()
    with tm:
        if key=='internvl':
            from src.models.internvl import InternChoiceModel
            model=InternChoiceModel(long_edge=768)
        else:
            pins=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
            model=ChoiceModel(pins['model_id'],pins['revision'],key,long_edge=768)
        for item in items:
            image=Image.open(item['canonical_path']).convert('RGB');task=item['dataset'];iid=item['image_id']
            for factor,operator,params,scale in [('gray128','occlusion',{'fill':'gray128','halo':0},1),('blur12','blur',{'sigma':12},1),('blur36','blur',{'sigma':36},1),('mask075','occlusion',{},.75),('mask125','occlusion',{},1.25)]:
                variants=[('clean',image)]
                for ri,box in enumerate(item['regions'][:2]):
                    edit=request(image,scaled_box(box,image.size,scale),operator,**params)
                    for condition in ['removal','sham']:variants.append((f'r{ri}_{condition}',Image.open(edit[condition+'_path']).convert('RGB')))
                for variant,im in variants:
                    check_window();cell=(task,iid,factor,variant)
                    if cell in done:continue
                    im,q,options,gold=present(im,item);row=model.score(im,q,options)
                    row.update(task=task,image_id=iid,factor=factor,variant=variant,gold=gold,correct=row['prediction']==gold,operator=operator,operator_params=params,mask_linear_scale=scale)
                    record['rows'].append(row);done.add(cell);atomic_json(out,record)
            # Full matched factorial at reduced resolution; letter rotation only on RefCOCOg.
            for factor,edge,style in [('resolution512',512,'legacy'),('outer_rotation',768,'outer_rotation')]:
                if factor=='outer_rotation' and task!='refcocog':continue
                model.set_long_edge(edge)
                variants=[('clean',item['canonical_path'])]
                for edit in edits:
                    if edit['task']==task and edit['image_id']==iid:
                        for condition in ['removal','sham']:variants.append((f"r{edit['region_index']}_{edit['operator']}_{condition}",edit[condition+'_path']))
                for variant,path in variants:
                    check_window();cell=(task,iid,factor,variant)
                    if cell in done:continue
                    im,q,options,gold=present(Image.open(path).convert('RGB'),item,style);row=model.score(im,q,options)
                    row.update(task=task,image_id=iid,factor=factor,variant=variant,gold=gold,correct=row['prediction']==gold)
                    record['rows'].append(row);done.add(cell);atomic_json(out,record)
                model.set_long_edge(768)
            update('F E4 real-image ablations',model=key,completed=len(done),last_image=iid)
    record.update(status='COMPLETE_EXPLORATORY',telemetry=tm.result,cumulative_budget=model.budget.value);atomic_json(out,record)
    update('F E4 real-image ablations',status='COMPLETE_EXPLORATORY',model=key,score_rows=len(done))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','internvl'],required=True);run(p.parse_args().model)
