"""Untuned same-color-type GQA diagnostic, separate from all v1 result files."""
import argparse,json
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.storage import atomic_json
from src.session import update,check_window


def run(key):
    path=Path(f'reports/evidence/v2/hard_gqa_{key}.json')
    result=json.loads(path.read_text()) if path.exists() else {'model':key,'rows':[],'status':'RUNNING','task':'GQA_color_focused_diagnostic','partition':'development_hard'}
    done={(r['image_id'],r['condition']) for r in result['rows']}
    update('C harder GQA scoring',model=key)
    tm=Telemetry()
    with tm:
        if key=='internvl':
            from src.models.internvl import InternChoiceModel
            model=InternChoiceModel()
        else:
            pin=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
            model=ChoiceModel(pin['model_id'],pin['revision'],key,long_edge=768)
        items=json.loads(Path('outputs/v2/gqa_hard_manifest.json').read_text())
        for i,item in enumerate(items):
            image=Image.open(Path('datasets')/item['image_path']).convert('RGB')
            shuffled=Image.open(Path('datasets')/items[(i+1)%len(items)]['image_path']).convert('RGB').resize(image.size)
            for c,im in [('image',image),('text_only',None),('shuffled',shuffled)]:
                check_window()
                if (item['image_id'],c) in done: continue
                row=model.score(im,item['question'],item['options'],verify=i==0 and c=='image')
                row.update(image_id=item['image_id'],item_id=item['item_id'],condition=c,gold=item['gold'],correct=row['prediction']==item['gold'])
                result['rows'].append(row); atomic_json(path,result)
    counts={c:sum(r['correct'] for r in result['rows'] if r['condition']==c) for c in ['image','text_only','shuffled']}
    result.update(status='COMPLETE',metrics={**counts,'n':len(items),'screen_passed':counts['image']/len(items)>=.4 and counts['image']>counts['text_only']},
                  telemetry=tm.result,cumulative_budget=model.budget.value)
    atomic_json(path,result); update('C harder GQA scoring',status='COMPLETE',model=key,metrics=result['metrics'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','smol','internvl'],required=True);run(p.parse_args().model)
