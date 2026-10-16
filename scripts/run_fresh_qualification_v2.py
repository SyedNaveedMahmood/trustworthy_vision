"""Fresh independent40-image qualification, fixed768/legacy, all three backbones."""
import argparse,json
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.storage import atomic_json,digest
from src.session import update,check_window


def run(key):
    tasks={t:json.loads(Path(f'outputs/manifests/{t}_qualification_corrected.json').read_text()) for t in ['gqa','refcocog']}
    mh=digest(tasks);path=Path(f'reports/evidence/v2/fresh_qualification_{key}.json');record=json.loads(path.read_text()) if path.exists() else {'rows':[],'status':'RUNNING','model':key,'manifest_sha256':mh,'edge':768,'style':'legacy','selection':'fixed before corrected qualification outcomes'}
    if record['manifest_sha256']!=mh:raise ValueError('Qualification manifest changed')
    done={(r['task'],r['image_id'],r['condition']) for r in record['rows']};tm=Telemetry()
    with tm:
        if key=='internvl':
            from src.models.internvl import InternChoiceModel
            model=InternChoiceModel(long_edge=768)
        else:
            pins=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
            model=ChoiceModel(pins['model_id'],pins['revision'],key,long_edge=768)
        for task,items in tasks.items():
            if len(items)!=20:raise ValueError('Incomplete qualification manifest')
            for index,item in enumerate(items):
                conditions=['image','text_only','shuffled'] if task=='gqa' else ['image','text_only']
                for condition in conditions:
                    check_window()
                    if (task,item['image_id'],condition) in done:continue
                    image=Image.open(Path('datasets')/(items[(index+1)%len(items)]['image_path'] if condition=='shuffled' else item['image_path'])).convert('RGB')
                    image,q,options,gold=present(image,item);row=model.score(image if condition!='text_only' else None,q,options,verify=task=='gqa' and index==0 and condition=='image')
                    row.update(task=task,image_id=item['image_id'],condition=condition,gold=gold,correct=row['prediction']==gold)
                    record['rows'].append(row);done.add((task,item['image_id'],condition));atomic_json(path,record)
                    if row['answer_step_total_label_mass']<.001 and not row['unrestricted_argmax_in_label_set']:
                        raise RuntimeError('Canonical answer continuation requires audit: '+row['answer_step_argmax_token'])
        record['metrics']={}
        for task in tasks:
            counts={condition:sum(r['correct'] for r in record['rows'] if r['task']==task and r['condition']==condition) for condition in ['image','text_only','shuffled'] if task=='gqa' or condition!='shuffled'}
            record['metrics'][task]={'n':20,**counts,'passed':counts['image']>=8 and counts['image']>counts['text_only']}
    record.update(status='COMPLETE',telemetry=tm.result,cumulative_budget=model.budget.value);atomic_json(path,record)
    update('A fresh corrected qualification',status='COMPLETE',model=key,metrics=record['metrics'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','smol','internvl'],required=True);run(p.parse_args().model)
