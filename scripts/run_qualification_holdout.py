"""Fixed 768px legacy presentation on disjoint qualification, no tuning after outputs."""
import argparse,json
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.storage import atomic_json,read_checkpoint
from src.session import update,check_window


def run(key):
    p=Path(f'reports/evidence/v2/holdout_{key}.json')
    record=json.loads(p.read_text()) if p.exists() else {'rows':[],'model':key,'edge':768,'presentation':'legacy','status':'RUNNING','partition':'qualification','selection_rule':'smallest development resolution meeting fixed 40pct screen, no presentation change'}
    done={(r['task'],r['image_id'],r['condition']) for r in record['rows']}
    pins=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'].get(key)
    tm=Telemetry(); update('A disjoint qualification',model=key)
    with tm:
        if key=='internvl':
            from src.models.internvl import InternChoiceModel
            model=InternChoiceModel(long_edge=768)
        else: model=ChoiceModel(pins['model_id'],pins['revision'],key,long_edge=768)
        for task in ['gqa','refcocog']:
            rows=json.loads(Path(f'outputs/manifests/{task}_qualification.json').read_text())
            if len(rows)!=20: raise ValueError('Incomplete qualification manifest')
            for item in rows:
                for condition in ['image','text_only']:
                    check_window()
                    if (task,item['image_id'],condition) in done: continue
                    im=Image.open(Path('datasets')/item['image_path']).convert('RGB')
                    im,q,options,gold=present(im,item)
                    score=model.score(im if condition=='image' else None,q,options)
                    score.update(task=task,image_id=item['image_id'],item_id=item['item_id'],condition=condition,gold=gold,correct=score['prediction']==gold)
                    record['rows'].append(score); atomic_json(p,record)
        record['metrics']={}
        for task in ['gqa','refcocog']:
            counts={c:sum(r['correct'] for r in record['rows'] if r['task']==task and r['condition']==c) for c in ['image','text_only']}
            record['metrics'][task]={**counts,'n':20,'passed':counts['image']>=8 and counts['image']>counts['text_only']}
    record.update(status='COMPLETE',telemetry=tm.result,cumulative_budget=model.budget.value)
    atomic_json(p,record); update('A disjoint qualification',status='COMPLETE',model=key,metrics=record['metrics'])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=['qwen','smol','internvl']); run(p.parse_args().model)
