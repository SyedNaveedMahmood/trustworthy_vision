"""Bounded official Smol last-logit optimization equivalence test; main runs unchanged."""
import json,time,inspect
from pathlib import Path
import numpy as np
import torch
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.models.choice import ChoiceModel,Telemetry
from src.presentation import present
from src.storage import atomic_json
from src.session import update,check_window


@torch.inference_mode()
def run():
    completed=Path('reports/evidence/v2/last_logits_optimization.json')
    if completed.exists() and json.loads(completed.read_text()).get('status')=='COMPLETE_NUMERICAL_OPTIMIZATION_DIAGNOSTIC':
        print('Completed optimization diagnostic preserved; zero new forwards')
        return
    check_window()
    budget=json.loads(Path('outputs/budget.json').read_text())
    if budget['forward_calls']+12>12000:raise RuntimeError('No reserved budget for complete optimization diagnostic')
    manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items'];items=[r for t in ['gqa','refcocog'] for r in [x for x in manifest if x['dataset']==t][:3]]
    pin=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models']['smol'];tm=Telemetry();rows=[]
    with tm:
        model=ChoiceModel(pin['model_id'],pin['revision'],'smol',long_edge=768)
        if 'logits_to_keep' not in inspect.signature(model.model.forward).parameters:raise RuntimeError('Pinned official model lacks requested last-logit optimization')
        for index,item in enumerate(items):
            check_window();image,q,options,gold=present(Image.open(item['canonical_path']).convert('RGB'),item);prompt,inputs,labels=model.encode(image,q,options)
            outputs={};measure={}
            for mode in (['full','last'] if index%2==0 else ['last','full']):
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter();before=model.budget.value
                result=model.budget.forward(model.model,**inputs,**({'logits_to_keep':1} if mode=='last' else {}))
                outputs[mode]=result.logits[0,-1,:].float().cpu().numpy().copy();del result
                after=model.budget.value
                measure[mode]={'wall_seconds':time.perf_counter()-start,'cuda_stream_seconds':after['cuda_stream_seconds']-before['cuda_stream_seconds'],
                               'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'logit_count':len(outputs[mode])}
            rows.append({'task':item['dataset'],'image_id':item['image_id'],'all_vocabulary_max_abs_logit_difference':float(abs(outputs['full']-outputs['last']).max()),
                         'choice_max_abs_logit_difference':float(abs(outputs['full'][labels]-outputs['last'][labels]).max()),'measurements':measure})
    summary={'status':'COMPLETE_NUMERICAL_OPTIMIZATION_DIAGNOSTIC','model':'smol','rows':rows,'forward_calls':12,'telemetry':tm.result,
        'all_logits_bit_identical':all(r['all_vocabulary_max_abs_logit_difference']==0 for r in rows),
        'median_cuda_stream_seconds':{mode:float(np.median([r['measurements'][mode]['cuda_stream_seconds'] for r in rows])) for mode in ['full','last']},
        'peak_allocated_bytes_by_mode':{mode:max(r['measurements'][mode]['torch_peak_allocated_bytes'] for r in rows) for mode in ['full','last']},
        'adoption':'Not applied retroactively to experiments. Official optional flag tested on six fixed real inputs only; future adoption requires recorded configuration.',
        'qwen':'Pinned4.51.3 Qwen implementation has no equivalent explicit logits_to_keep parameter; no unverified custom replacement attempted.'}
    atomic_json('reports/evidence/v2/last_logits_optimization.json',summary);update('G official last-logit optimization test',status='COMPLETE',all_logits_bit_identical=summary['all_logits_bit_identical'])


if __name__=='__main__':run()
