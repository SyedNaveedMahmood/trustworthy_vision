"""No-oracle evidence candidates/anchor; fixed-grid feasibility, not causal reliability."""
import json,collections
from pathlib import Path
import numpy as np
from src.scoring.v2 import estimators
from src.storage import atomic_json
from src.session import update


def risk_curve(correct,scores):
    # Stable image order breaks ties; report ties rather than implying unique ranks.
    correct=np.asarray(correct,dtype=float);scores=np.asarray(scores,dtype=float);order=np.argsort(-scores,kind='stable')
    risk=(1-correct[order]).cumsum()/np.arange(1,len(correct)+1)
    return {'aurc':float(risk.mean()),'risk_at_coverage':{str(c):float(risk[max(0,int(np.ceil(c*len(correct)))-1)]) for c in [.5,.7,.85,.95,1]},
            'n_score_ties':len(scores)-len(set(scores.tolist())),'risk_curve':risk.tolist(),'coverage':(np.arange(1,len(correct)+1)/len(correct)).tolist()}


def run():
    result={'status':'NO_ORACLE_EVIDENCE_DIAGNOSTIC','human_audit':'HUMAN_AUDIT_PENDING','cells':{},'items':[],
            'limitations':['Grid candidate generation uses image dimensions only; anchor is fixed clean predicted answer.',
                          'The image cohort was selected using annotation-based development eligibility; it is not an annotation-free image sampler.',
                          'RefCOCOg QA answer boxes remain the existing task input. This does not establish end-to-end annotation-free localization.',
                          'No target/negative ranking accuracy is defined for unlabeled grid regions. Max sensitivity is not verified reliability.']}
    for model in ['qwen','internvl']:
        record=json.loads(Path(f'reports/evidence/v2/E2_grid_corrected_{model}.json').read_text())
        if record['status']!='COMPLETE_COMPUTATIONAL_ONLY':raise ValueError('Incomplete grid scoring')
        groups=collections.defaultdict(dict)
        for r in record['rows']:groups[(r['task'],r['image_id'])][r['variant']]=r
        for (task,iid),g in sorted(groups.items()):
            if len(g)!=49:raise ValueError('Grid factorial incomplete')
            clean=g['clean'];idx='ABCD'.index(clean['prediction']);regions=[]
            for ri in range(4):
                arrays=[[g[f'r{r}_{op}_{c}']['probabilities'][idx] for op in ['occlusion','blur','lama']] for r,c in [(ri,'removal'),(ri,'sham'),(ri+4,'removal'),(ri+4,'sham')]]
                regions.append(estimators(clean['probabilities'][idx],*arrays))
            p=np.asarray(clean['probabilities']);entropy=float(-np.sum(p*np.log(np.maximum(p,1e-12))))
            result['items'].append({'task':task,'model':model,'image_id':iid,'original_correct':clean['correct'],'anchor_letter':clean['prediction'],
                                    'confidence':max(clean['probabilities']),'negative_entropy':-entropy,'regions':regions,
                                    'max_robust_sham':max(r['robust_sham'] for r in regions),'max_robust_independent':max(r['robust_independent'] for r in regions)})
        for task in ['gqa','refcocog']:
            rows=[r for r in result['items'] if r['model']==model and r['task']==task]
            result['cells'][f'{task}/{model}']={'n':len(rows),'correct':sum(r['original_correct'] for r in rows),
                'methods':{method:risk_curve([r['original_correct'] for r in rows],[r[method] for r in rows]) for method in ['confidence','negative_entropy','max_robust_sham','max_robust_independent']},
                'theoretical_scores_per_image':49,'confidence_scores_per_image':1,'no_independently_validated_region_relevance':True}
    atomic_json('reports/evidence/v2/no_oracle_grid_analysis.json',result)
    update('E annotation-free grid analysis',status='COMPLETE_EXPLORATORY',images_per_model=16)


if __name__=='__main__':run()
