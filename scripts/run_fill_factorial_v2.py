"""Explicit 2x2 neutral-fill x halo factorial; reuses completed cached forwards."""
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


def run(key):
    manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items']
    items=[r for task in ['gqa','refcocog'] for r in [x for x in manifest if x['dataset']==task][:6]]
    out=Path(f'reports/evidence/v2/E4_fill_factorial_{key}.json')
    record=json.loads(out.read_text()) if out.exists() else {'rows':[],'model':key,'status':'RUNNING','manifest_sha256':digest(items),'design':'2x2_fill_mean_gray128_by_halo_0_8','human_audit':'HUMAN_AUDIT_PENDING'}
    if record['manifest_sha256']!=digest(items):raise ValueError('Manifest changed')
    done={(r['task'],r['image_id'],r['factor'],r['variant']) for r in record['rows']};tm=Telemetry()
    with tm:
        if key=='internvl':
            from src.models.internvl import InternChoiceModel
            model=InternChoiceModel(long_edge=768)
        else:
            p=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
            model=ChoiceModel(p['model_id'],p['revision'],key,long_edge=768)
        for item in items:
            im=Image.open(item['canonical_path']).convert('RGB')
            for fill in ['mean','gray128']:
                for halo in [0,8]:
                    factor=f'{fill}_halo{halo}';variants=[('clean',im)]
                    for ri,box in enumerate(item['regions'][:2]):
                        edit=request(im,box,'occlusion',fill=fill,halo=halo)
                        for c in ['removal','sham']:variants.append((f'r{ri}_{c}',Image.open(edit[c+'_path']).convert('RGB')))
                    for variant,image in variants:
                        check_window();cell=(item['dataset'],item['image_id'],factor,variant)
                        if cell in done:continue
                        image,q,options,gold=present(image,item);row=model.score(image,q,options)
                        row.update(task=item['dataset'],image_id=item['image_id'],factor=factor,variant=variant,gold=gold,correct=row['prediction']==gold,options=options)
                        record['rows'].append(row);done.add(cell);atomic_json(out,record)
            update('F E4 fill-halo factorial',model=key,completed=len(done))
    record.update(status='COMPLETE_EXPLORATORY',telemetry=tm.result,cumulative_budget=model.budget.value);atomic_json(out,record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','internvl'],required=True);run(p.parse_args().model)
