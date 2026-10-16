"""Preserve initial/extension/pooled diagnostics across all three qualified backbones."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.analyze_interventions_v2 import extract,METHODS,ABLATIONS
from src.evaluation.v2 import paired_summary,pair_credit
from src.evaluation import paired_cluster_bootstrap
from src.storage import atomic_json
from src.session import update

BASELINES=['raw_occlusion','blur_only','lama_only','raw_mean','raw_median','sham_mean','sham_median','independent_control_mean','independent_control_median']
COSTS={**{m:3 for m in ['raw_occlusion','blur_only','lama_only']},**{m:7 for m in ['raw_mean','raw_median']},**{m:13 for m in ['sham_mean','sham_median']},**{m:25 for m in ['independent_control_mean','independent_control_median']}}


def run(initial_only=False):
    cohorts=['corrected'] if initial_only else ['corrected','extension'];models=['qwen','smol','internvl'];data={};source={}
    for cohort in cohorts:
        data[cohort]={}
        for model in models:
            path=Path(f'reports/evidence/v2/E2_pairs_{cohort}_{model}.json');record=json.loads(path.read_text())
            if record['status']!='COMPLETE_COMPUTATIONAL_ONLY':raise ValueError(f'Incomplete {cohort}/{model}; no silent missing-image deletion')
            rows=extract(record);data[cohort][model]=rows;source[cohort+'/'+model]={'rows':len(record['rows']),'manifest_sha256':record['manifest_sha256'],'telemetry':record['telemetry']}
        ids=[{(r['task'],r['image_id'],r['anchor']) for r in data[cohort][m]} for m in models]
        if any(s!=ids[0] for s in ids):raise ValueError('Incomplete shared-model factorial')
    if not initial_only:
        for model in models:
            a={(r['task'],r['image_id']) for r in data['corrected'][model]};b={(r['task'],r['image_id']) for r in data['extension'][model]}
            if a&b:raise ValueError('Extension image leakage')
        data['pooled']={m:data['corrected'][m]+data['extension'][m] for m in models}
    result={'status':'EXPLORATORY_NOT_CONFIRMATORY','primary_backbones':['qwen','smol'],'third_backbone':'internvl','human_audit':'HUMAN_AUDIT_PENDING','cells':{},'joint_contrasts':{},'sources':source,'details':data,
            'note':'Corrected Smol canonical space-prefixed continuations. Color-only GQA; annotation-reference region ranking. Initial and extension remain separate before pooled descriptive summaries.'}
    lines=['| Cohort | Task | Model | Anchor | Method | n | Ranking (95% paired bootstrap CI) | Pooled AUROC |','|---|---|---|---|---|---:|---|---:|']
    for cohort,model_rows in data.items():
        for model,rows in model_rows.items():
            for task in ['gqa','refcocog']:
                for anchor in ['gold','predicted']:
                    selected=[r for r in rows if r['task']==task and r['anchor']==anchor];key=f'{cohort}/{task}/{model}/{anchor}';result['cells'][key]={}
                    for method in METHODS+ABLATIONS:
                        stats=paired_summary([r['positive'][method] for r in selected],[r['negative'][method] for r in selected]);result['cells'][key][method]=stats
                        if method in METHODS:
                            lo,hi=stats['ranking_ci95'];lines.append(f"| {cohort} | {task} | {model} | {anchor} | {method} | {stats['n']} | {stats['ranking_accuracy']:.3f} [{lo:.3f},{hi:.3f}] | {stats['pooled_region_auroc']:.3f} |")
        for pair in [['qwen','smol'],['qwen','internvl']]:
            for anchor in ['gold','predicted']:
                for baseline in METHODS:
                    rows=[{'task':r['task'],'model':m,'image_id':r['image_id'],'candidate_correct':pair_credit(r['positive']['robust_independent'],r['negative']['robust_independent']),
                           'baseline_correct':pair_credit(r['positive'][baseline],r['negative'][baseline])} for m in pair for r in model_rows[m] if r['anchor']==anchor]
                    result['joint_contrasts'][f"{cohort}/{'_'.join(pair)}/{anchor}/{baseline}"]=paired_cluster_bootstrap(rows,5000,20269)
    if 'pooled' in data:
        performance={method:float(np.mean([result['cells'][f'pooled/{t}/{m}/gold'][method]['ranking_accuracy'] for t in ['gqa','refcocog'] for m in ['qwen','smol']])) for method in BASELINES}
        best=sorted(BASELINES,key=lambda m:(-performance[m],COSTS[m],m))[0]
        result['comparator_proposal']={'status':'DEVELOPMENT_SELECTED_NOT_LOCKED','method':best,'macro_ranking_by_method':performance,'direct_scores_per_pair':COSTS[best],
            'selection_rule':'largest gold-anchor macro pair accuracy on pooled development, original intended Qwen/Smol; ties lower direct call cost, then lexical method ID',
            'candidate_vs_selected_baseline':result['joint_contrasts'][f'pooled/qwen_smol/gold/{best}']}
    name='E3_corrected_initial' if initial_only else 'E3_replication'
    atomic_json(f'reports/evidence/v2/{name}.json',result);Path(f'reports/evidence/v2/{name}_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    update('E corrected three-backbone comparison',status='COMPLETE_EXPLORATORY',cohorts=list(data),models=models)
    print(json.dumps(result.get('comparator_proposal',{'status':'initial_only'}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial-only',action='store_true');a=p.parse_args();run(a.initial_only)
