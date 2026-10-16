"""Descriptive E4 analyses, preserving each paired image and fixed coefficient."""
import json,collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.analyze_interventions_v2 import extract,OPS,METHODS,ABLATIONS
from src.scoring import bounded_logit
from src.evaluation.v2 import paired_summary,pair_credit,paired_difference
from src.scoring.paper_baselines import ced
from src.storage import atomic_json
from src.session import update


def logit(p):return float(bounded_logit(p)[0])


def run():
    e3=json.loads(Path('reports/evidence/v2/E3_comparison.json').read_text());result={'status':'EXPLORATORY','human_audit':'HUMAN_AUDIT_PENDING','operator_diagnostics':{},'numerical_ablations':{},'image_ablations':{},'fill_factorial':{}}
    lines=['| Task/model | Factor | Anchor | n | Pair ranking (95% CI) |','|---|---|---|---:|---|']
    for model,details in e3['details'].items():
        record=json.loads(Path(f'reports/evidence/v2/E4_{model}.json').read_text())
        if record['status']!='COMPLETE_EXPLORATORY':raise ValueError('E4 image run incomplete')
        for task in ['gqa','refcocog']:
            for anchor in ['gold','predicted']:
                base=[r for r in details if r['task']==task and r['anchor']==anchor];key=f'{task}/{model}/{anchor}'
                result['numerical_ablations'][key]={method:e3['cells'][key][method] for method in METHODS+ABLATIONS}
                regions=[r[role] for r in base for role in ['positive','negative']]
                effects=np.array([r['raw_effects'] for r in regions]);shams=np.array([r['sham_effects'] for r in regions])
                inversion=[]
                for row in base:
                    diff=np.array(row['positive']['raw_effects'])-np.array(row['negative']['raw_effects'])
                    inversion.append(any(diff[i]*diff[j]<0 for i in range(3) for j in range(i)))
                result['operator_diagnostics'][key]={'mean_operator_std':float(effects.std(axis=1).mean()),'pair_any_operator_rank_inversion_rate':float(np.mean(inversion)),
                    'sham_abs_effect_gt_1_rate':float(np.mean(abs(shams)>1)), 'sham_abs_effect_mean':float(abs(shams).mean()),
                    'positive_mean_scores':{method:float(np.mean([r['positive'][method] for r in base])) for method in ['raw_mean','sham_median','robust_sham','robust_independent']},
                    'negative_mean_scores':{method:float(np.mean([r['negative'][method] for r in base])) for method in ['raw_mean','sham_median','robust_sham','robust_independent']},
                    'operator_spearman':{f'{OPS[i]}/{OPS[j]}':float(spearmanr(effects[:,i],effects[:,j]).statistic) if np.std(effects[:,i]) and np.std(effects[:,j]) else None for i in range(3) for j in range(i)}}
                for factor in sorted({r['factor'] for r in record['rows'] if r['task']==task}):
                    rows=[r for r in record['rows'] if r['task']==task and r['factor']==factor]
                    p=[];n=[];ids=[]
                    if factor in ['resolution512','outer_rotation']:
                        out=extract({'rows':rows});out=[r for r in out if r['anchor']==anchor]
                        for r in out:p.append(r['positive']['robust_sham']);n.append(r['negative']['robust_sham']);ids.append(r['image_id'])
                        method='robust_sham'
                    else:
                        method='single_operator_sham'
                        grouped=collections.defaultdict(dict)
                        for r in rows:grouped[r['image_id']][r['variant']]=r
                        for iid,g in sorted(grouped.items()):
                            idx='ABCD'.index(g['clean']['gold'] if anchor=='gold' else g['clean']['prediction'])
                            vals=[logit(g[f'r{ri}_sham']['probabilities'][idx])-logit(g[f'r{ri}_removal']['probabilities'][idx]) for ri in [0,1]]
                            p.append(vals[0]);n.append(vals[1]);ids.append(iid)
                    stats=paired_summary(p,n);result['image_ablations'][key+'/'+factor]={'method':method,'image_ids':ids,**stats}
                    lo,hi=stats['ranking_ci95'];lines.append(f"| {key} | {factor} ({method}) | {anchor} | {stats['n']} | {stats['ranking_accuracy']:.3f} [{lo:.3f},{hi:.3f}] |")
        factorial=Path(f'reports/evidence/v2/E4_fill_factorial_{model}.json')
        if factorial.exists():
            f=json.loads(factorial.read_text())
            if f['status']!='COMPLETE_EXPLORATORY':raise ValueError('Fill factorial incomplete')
            for task in ['gqa','refcocog']:
                for factor in sorted({r['factor'] for r in f['rows']}):
                    groups=collections.defaultdict(dict)
                    for r in f['rows']:
                        if r['task']==task and r['factor']==factor:groups[r['image_id']][r['variant']]=r
                    for method in ['raw','sham','ced','ced_soft']:
                        values=[]
                        for iid,g in sorted(groups.items()):
                            clean=g['clean'];idx='ABCD'.index(clean['gold']);vals=[]
                            for ri in [0,1]:
                                removal=g[f'r{ri}_removal'];sham=g[f'r{ri}_sham']
                                if method in ['ced','ced_soft']:
                                    options=clean['options'];a=options['ABCD'.index(clean['prediction'])];b=options['ABCD'.index(removal['prediction'])]
                                    value=ced(a,b)[method]
                                else:value=logit((clean if method=='raw' else sham)['probabilities'][idx])-logit(removal['probabilities'][idx])
                                vals.append(value)
                            values.append(vals)
                        result['fill_factorial'][f'{task}/{model}/{factor}/{method}']=paired_summary(*np.array(values).T)
    atomic_json('reports/evidence/v2/E4_analysis.json',result)
    Path('reports/evidence/v2/E4_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Failures sorted by predeclared raw/robust pair disagreement, not removed.
    failures=[]
    manifest=json.loads(Path('reports/evidence/v2/pilot_sample_summary_corrected.json').read_text())['items']
    for model,rows in e3['details'].items():
        for r in rows:
            if r['anchor']!='gold':continue
            a=pair_credit(r['positive']['robust_independent'],r['negative']['robust_independent']);b=pair_credit(r['positive']['raw_occlusion'],r['negative']['raw_occlusion'])
            if a<1 or a!=b:
                item=next(x for x in manifest if x['dataset']==r['task'] and x['image_id']==r['image_id'])
                failures.append({'model':model,'task':r['task'],'image_id':r['image_id'],'robust_pair_credit':a,'occlusion_pair_credit':b,'original_correct':r['original_correct'],
                                 'area_mismatch':item['area_mismatch'],'control_opposite_overlap':item['controls_overlap_opposite_reference'],
                                 'raw_positive_negative':[r['positive']['raw_effects'],r['negative']['raw_effects']],
                                 'corrected_positive_negative':[r['positive']['independent_effects'],r['negative']['independent_effects']]})
    atomic_json('reports/evidence/v2/E4_failure_cases.json',failures)
    update('F E4 analysis',status='COMPLETE_EXPLORATORY',failure_cases=len(failures))


if __name__=='__main__':run()
