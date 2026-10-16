"""Audit exact answer-step vocabulary mass, without generating or tuning prompts."""
import argparse,json
from pathlib import Path
import numpy as np
import truststore
truststore.inject_into_ssl()
import torch
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.storage import atomic_json,digest
from src.session import update,check_window


@torch.inference_mode()
def run(key):
    pins=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
    items=json.loads(Path('outputs/manifests/refcocog_development.json').read_text())[:20]
    path=Path(f'reports/evidence/v2/choice_mass_{key}.json');record=json.loads(path.read_text()) if path.exists() else {'rows':[],'model':key,'status':'RUNNING','selection':'first20_original_development_images_no_outcome_filter','manifest_sha256':digest(items)}
    if record['manifest_sha256']!=digest(items):raise ValueError('Manifest changed')
    done={(r['image_id'],r['condition']) for r in record['rows']};tm=Telemetry()
    with tm:
        model=ChoiceModel(pins['model_id'],pins['revision'],key)
        for edge,style in [(512,'legacy'),(768,'legacy'),(768,'numbered'),(768,'outer_rotation')]:
            model.set_long_edge(edge);condition=f'{style}_{edge}_image'
            for item in items:
                check_window()
                if (item['image_id'],condition) in done:continue
                image=Image.open(Path('datasets')/item['image_path']).convert('RGB');image,q,options,gold=present(image,item,style)
                text,inputs,labels=model.encode(image,q,options);logits=model.forward_inputs(inputs).logits[0,-1,:].float()
                logp=torch.log_softmax(logits.double(),dim=-1);choice_mass=float(logp[labels].exp().sum().cpu());choice=logits[labels].cpu().numpy();prob=np.exp(choice-choice.max());prob/=prob.sum()
                top=torch.topk(logp,5);top_ids=top.indices.cpu().tolist();best=int(logits.argmax().cpu())
                record['rows'].append({'image_id':item['image_id'],'condition':condition,'gold':gold,'prediction':'ABCD'[int(prob.argmax())],
                  'probabilities':prob.tolist(),'choice_logits':choice.tolist(),'four_label_vocabulary_mass':choice_mass,
                  'unrestricted_argmax_is_label':best in labels,'unrestricted_argmax_token':model.processor.tokenizer.convert_ids_to_tokens(best),
                  'top5_tokens':model.processor.tokenizer.convert_ids_to_tokens(top_ids),'top5_probabilities':top.values.exp().cpu().tolist(),
                  'label_ids':labels,'input_preprocessing':model.last_preprocess})
                done.add((item['image_id'],condition));atomic_json(path,record)
        record['summary']={c:{'n':len(rows),'mean_label_mass':float(np.mean([r['four_label_vocabulary_mass'] for r in rows])),
            'min_label_mass':min(r['four_label_vocabulary_mass'] for r in rows),'greedy_label_fraction':float(np.mean([r['unrestricted_argmax_is_label'] for r in rows]))}
            for c in sorted({r['condition'] for r in record['rows']}) for rows in [[r for r in record['rows'] if r['condition']==c]]}
    record.update(status='COMPLETE_DIAGNOSTIC',telemetry=tm.result,cumulative_budget=model.budget.value);atomic_json(path,record)
    update('A full-vocabulary choice audit',status='COMPLETE',model=key,summary=record['summary'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','smol'],required=True);run(p.parse_args().model)
