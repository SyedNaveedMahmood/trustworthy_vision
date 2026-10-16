"""Corrected canonical answer-continuation development audit; old results immutable."""
import json
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.storage import atomic_json
from src.session import update,check_window


def run():
    path=Path('reports/evidence/v2/qualification_corrected_smol.json');record=json.loads(path.read_text()) if path.exists() else {'rows':[],'status':'RUNNING','canonical_label_separator':' ','cohort':'original40_development_grounding'}
    done={(r['image_id'],r['condition']) for r in record['rows']};tm=Telemetry()
    pins=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models']['smol']
    with tm:
        model=ChoiceModel(pins['model_id'],pins['revision'],'smol')
        for edge,style in [(512,'legacy'),(768,'legacy'),(768,'numbered'),(768,'outer_rotation'),(768,'visual_rotation')]:
            model.set_long_edge(edge);condition=f'{style}_{edge}_image'
            for index,item in enumerate(json.loads(Path('outputs/manifests/refcocog_development.json').read_text())):
                check_window()
                if (item['image_id'],condition) in done:continue
                image=Image.open(Path('datasets')/item['image_path']).convert('RGB');image,q,options,gold=present(image,item,style)
                row=model.score(image,q,options,verify=index==0 and edge==512)
                row.update(task='refcocog',image_id=item['image_id'],condition=condition,gold=gold,correct=row['prediction']==gold)
                record['rows'].append(row);done.add((item['image_id'],condition));atomic_json(path,record)
            update('A corrected Smol development',condition=condition,correct=sum(r['correct'] for r in record['rows'] if r['condition']==condition))
    record.update(status='COMPLETE_DEVELOPMENT_ONLY',telemetry=tm.result,cumulative_budget=model.budget.value);atomic_json(path,record)


if __name__=='__main__':run()
