"""E3/E4 analysis of complete cached probability vectors; no model forwards."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from src.scoring.v2 import estimators
from src.evaluation.v2 import paired_summary,pair_credit,paired_difference
from src.evaluation import paired_cluster_bootstrap
from src.storage import atomic_json
from src.session import update

OPS=['occlusion','blur','lama']
METHODS=['raw_occlusion','blur_only','lama_only','raw_mean','raw_median','sham_mean','sham_median','robust_sham','independent_control_mean','independent_control_median','robust_independent','offtarget_without_sham']
ABLATIONS=['loo_0','loo_1','loo_2','lambda_0','lambda_0.25','lambda_0.5','lambda_1']


def extract(record):
    groups=collections.defaultdict(dict)
    for row in record['rows']:
        key=(row['task'],row['image_id'])
        if row['variant'] in groups[key]: raise ValueError('Duplicate scoring cell')
        if len(row['probabilities'])!=4 or not np.isclose(sum(row['probabilities']),1) or not np.isfinite(row['probabilities']).all(): raise ValueError('Invalid probability vector')
        groups[key][row['variant']]=row
    out=[]
    for (task,iid),rows in sorted(groups.items()):
        clean=rows['clean']; expected=1+4*3*2
        if len(rows)!=expected: raise ValueError(f'Incomplete image factorial {task}/{iid}: {len(rows)}/{expected}')
        for anchor in ['gold','predicted']:
            label=clean['gold'] if anchor=='gold' else clean['prediction'];idx='ABCD'.index(label)
            regions=[]
            for ri in [0,1]:
                arrays=[]
                for rindex,condition in [(ri,'removal'),(ri,'sham'),(ri+2,'removal'),(ri+2,'sham')]:
                    arrays.append([rows[f'r{rindex}_{op}_{condition}']['probabilities'][idx] for op in OPS])
                regions.append(estimators(clean['probabilities'][idx],*arrays))
            out.append({'task':task,'image_id':iid,'anchor':anchor,'anchor_letter':label,'original_correct':clean['correct'],
                        'original_confidence':max(clean['probabilities']),'positive':regions[0],'negative':regions[1]})
    return out


def run():
    records={m:json.loads(Path(f'reports/evidence/v2/E2_pairs_corrected_{m}.json').read_text()) for m in ['qwen','internvl']}
    if any(r['status']!='COMPLETE_COMPUTATIONAL_ONLY' for r in records.values()): raise ValueError('Complete scores required')
    extracted={m:extract(r) for m,r in records.items()}
    if any({(r['task'],r['image_id'],r['anchor']) for r in rows}!={(r['task'],r['image_id'],r['anchor']) for r in extracted['qwen']} for rows in extracted.values()): raise ValueError('Shared image mismatch')
    result={'status':'EXPLORATORY_COMPUTATIONAL_ONLY','human_audit':'HUMAN_AUDIT_PENDING','cells':{},'paired_effects':{},'details':extracted,
            'note':'Pairwise AUROC is within-image concordance. Pooled AUROC compares all positive/negative regions with image-cluster bootstrap. Negative thresholds are diagnostics, not validated causal false-positive rates.'}
    table=['| Task | Model | Anchor | Method | n | Ranking (95% CI) | Pooled AUROC | Negative >1 |','|---|---|---|---|---:|---|---:|---:|']
    for anchor in ['gold','predicted']:
        for task in ['gqa','refcocog']:
            for model,rows in extracted.items():
                subset=[r for r in rows if r['task']==task and r['anchor']==anchor]
                key=f'{task}/{model}/{anchor}';result['cells'][key]={}
                for method in METHODS+ABLATIONS:
                    stats=paired_summary([r['positive'][method] for r in subset],[r['negative'][method] for r in subset])
                    result['cells'][key][method]=stats
                    if method in METHODS:
                        lo,hi=stats['ranking_ci95'];table.append(f"| {task} | {model} | {anchor} | {method} | {stats['n']} | {stats['ranking_accuracy']:.3f} [{lo:.3f}, {hi:.3f}] | {stats['pooled_region_auroc']:.3f} | {stats['negative_score_gt_1_rate']:.3f} |")
                # Descriptive paired contrasts, no comparator selected after observing effect.
                for baseline in METHODS:
                    a=[pair_credit(r['positive']['robust_independent'],r['negative']['robust_independent']) for r in subset]
                    b=[pair_credit(r['positive'][baseline],r['negative'][baseline]) for r in subset]
                    result['paired_effects'][key+'/'+baseline]=paired_difference(a,b)
        for baseline in ['raw_occlusion','raw_mean','sham_median','robust_sham']:
            paired=[{'task':r['task'],'model':m,'image_id':r['image_id'],
                     'candidate_correct':pair_credit(r['positive']['robust_independent'],r['negative']['robust_independent']),
                     'baseline_correct':pair_credit(r['positive'][baseline],r['negative'][baseline])} for m,rows in extracted.items() for r in rows if r['anchor']==anchor]
            result['paired_effects'][f'joint/{anchor}/{baseline}']=paired_cluster_bootstrap(paired,5000,20269)
    # Fixed descriptive strata, not eligibility filters.
    result['strata']={}
    for model,rows in extracted.items():
        for task in ['gqa','refcocog']:
            for correct in [True,False]:
                subset=[r for r in rows if r['task']==task and r['anchor']=='gold' and r['original_correct']==correct]
                result['strata'][f'{model}/{task}/correct={correct}']={method:{'n':len(subset),'ranking_accuracy':float(np.mean([pair_credit(r['positive'][method],r['negative'][method]) for r in subset])) if subset else None} for method in ['raw_occlusion','sham_median','robust_sham','robust_independent']}
    atomic_json('reports/evidence/v2/E3_comparison.json',result)
    Path('reports/evidence/v2/E3_tables.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    methods=['raw_occlusion','blur_only','lama_only','sham_median','robust_sham','robust_independent']
    for ax,task in zip(axes,['gqa','refcocog']):
        for k,m in enumerate(['qwen','internvl']):
            cells=result['cells'][f'{task}/{m}/gold'];xs=np.arange(len(methods))+(k-.5)*.15
            ax.plot(xs,[cells[x]['ranking_accuracy'] for x in methods],'o-',label=m)
        ax.set_xticks(range(len(methods)),methods,rotation=30,ha='right');ax.set_title(task);ax.axhline(.5,color='gray',linestyle='--');ax.set_ylim(0,1);ax.legend()
    axes[0].set_ylabel('Oracle-region pair ranking accuracy');fig.suptitle('Exploratory development; human edit validity pending');fig.tight_layout();fig.savefig('reports/evidence/v2/E3_ranking.png',dpi=150);plt.close(fig)
    update('E E3 cached-score analysis',status='COMPLETE_EXPLORATORY',paired_images_per_model=31)
    print(json.dumps({k:v for k,v in result['paired_effects'].items() if k.startswith('joint/')},indent=2))


if __name__=='__main__': run()
