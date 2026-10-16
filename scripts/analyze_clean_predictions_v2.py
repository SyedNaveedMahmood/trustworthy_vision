"""Clean calibration and oracle-region selective diagnostics after complete replication."""
import json
from pathlib import Path
import numpy as np
from src.evaluation.selectivity import risk_coverage,paired_aurc_bootstrap
from src.storage import atomic_json


def run():
    comparison=json.loads(Path('reports/evidence/v2/E3_replication.json').read_text());details=comparison['details']['pooled'];result={'cells':{},'joint_aurc':{},'status':'EXPLORATORY','scope':'Conditional four-option clean prediction calibration; sensitivity selectors use oracle-provided region pairs even with predicted answer anchor. Not deployment reliability.'}
    raw={}
    for model in ['qwen','smol','internvl']:
        for cohort in ['corrected','extension']:
            record=json.loads(Path(f'reports/evidence/v2/E2_pairs_{cohort}_{model}.json').read_text())
            for row in record['rows']:
                if row['variant']=='clean':raw[model,row['task'],row['image_id']]=row
        for task in ['gqa','refcocog']:
            selected=sorted([r for r in details[model] if r['task']==task and r['anchor']=='predicted'],key=lambda r:r['image_id'])
            clean=[raw[model,task,r['image_id']] for r in selected];p=np.array([r['probabilities'] for r in clean]);correct=np.array([r['correct'] for r in clean],dtype=int);conf=p.max(axis=1)
            gold=np.array(['ABCD'.index(r['gold']) for r in clean]);onehot=np.eye(4)[gold];bins=[];ece=0.
            for a,b in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
                mask=(conf>=a)&(conf<b if b<1 else conf<=b);n=int(mask.sum());acc=float(correct[mask].mean()) if n else None;avg=float(conf[mask].mean()) if n else None
                if n:ece+=n/len(conf)*abs(acc-avg)
                bins.append({'lower':float(a),'upper':float(b),'n':n,'accuracy':acc,'confidence':avg})
            scores={'confidence':conf.tolist(),'negative_entropy':np.sum(p*np.log(np.maximum(p,1e-12)),axis=1).tolist()}
            for method in ['raw_mean','sham_median','robust_sham','robust_independent']:
                scores['oracle_regions_max_'+method]=[max(r['positive'][method],r['negative'][method]) for r in selected]
            result['cells'][f'{task}/{model}']={'n':len(p),'accuracy':float(correct.mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(p)),gold],1e-12)).mean()),
                'multiclass_brier':float(np.sum((p-onehot)**2,axis=1).mean()),'ece_5_equal_width':float(ece),'calibration_bins':bins,
                'selection':{method:risk_coverage(correct,score) for method,score in scores.items()}}
    for method in ['raw_mean','sham_median','robust_sham','robust_independent']:
        rows=[]
        for model in ['qwen','smol']:
            for r in details[model]:
                if r['anchor']=='predicted':rows.append({'task':r['task'],'model':model,'image_id':r['image_id'],'correct':int(r['original_correct']),
                    'candidate_score':max(r['positive'][method],r['negative'][method]),'baseline_score':r['original_confidence']})
        result['joint_aurc'][method]=paired_aurc_bootstrap(rows)
    atomic_json('reports/evidence/v2/clean_calibration_and_oracle_selectivity.json',result)


if __name__=='__main__':run()
