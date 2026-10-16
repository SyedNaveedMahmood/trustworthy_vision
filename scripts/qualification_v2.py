"""One-factor development diagnostics; all original files remain immutable."""
import argparse,json,time
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.storage import atomic_json,digest,read_checkpoint
from src.session import update,check_window


def run(key):
    if key=='smol':raise RuntimeError('Historical bare-token Smol runner requires archived33720e1; use scripts.qualification_corrected_smol with current code. Do not mix scoring versions.')
    pins=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
    out=Path(f'reports/evidence/v2/qualification_{key}.json')
    record=json.loads(out.read_text()) if out.exists() else {'model':key,'status':'RUNNING','rows':[],'experiment':'development_resolution_presentation_diagnostics','selected_on_development':True}
    for row in record['rows']:
        # Recover immutable per-call settings if a live processor dictionary was reused.
        saved=read_checkpoint(Path('outputs/cache')/f"{row['run_key']}.json",row['run_key'])
        row['processor_config']=saved['processor_config']
    done={(r['image_id'],r['condition']) for r in record['rows']}
    update('A model diagnostics',model=key,completed_rows=len(done))
    tm=Telemetry()
    with tm:
        model=ChoiceModel(pins['model_id'],pins['revision'],key)
        items=json.loads(Path('outputs/manifests/refcocog_development.json').read_text())
        for edge,style,mode in [(512,'legacy','image'),(768,'legacy','image'),(1024,'legacy','image'),
                                 (768,'numbered','image'),(768,'outer_rotation','image'),(768,'visual_rotation','image'),
                                 (768,'numbered','text_only')]:
            model.set_long_edge(edge); condition=f'{style}_{edge}_{mode}'
            for i,item in enumerate(items):
                check_window()
                if (item['image_id'],condition) in done: continue
                image=Image.open(Path('datasets')/item['image_path']).convert('RGB')
                rendered,q,options,gold=present(image,item,style)
                row=model.score(rendered if mode=='image' else None,q,options)
                row.update(image_id=item['image_id'],item_id=item['item_id'],condition=condition,gold=gold,
                           correct=row['prediction']==gold,task='refcocog',partition='development')
                record['rows'].append(row); done.add((item['image_id'],condition)); atomic_json(out,record)
            update('A model diagnostics',model=key,condition=condition,completed_rows=len(done))
    record.update(status='COMPLETE_DEVELOPMENT_ONLY',telemetry=tm.result,cumulative_budget=model.budget.value)
    atomic_json(out,record)
    print(key,{c:sum(r['correct'] for r in record['rows'] if r['condition']==c) for c in sorted({r['condition'] for r in record['rows']})},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=['qwen','smol']); run(p.parse_args().model)
