"""Explicit fixed-choice inference budgets for cached exploratory estimators."""
import json
from pathlib import Path
import numpy as np
from src.evaluation.v2 import paired_summary
from src.scoring.v2 import robust
from src.storage import atomic_json


def run():
    original=json.loads(Path('reports/evidence/v2/E3_comparison.json').read_text());out={}
    costs={'raw_occlusion':3,'blur_only':3,'lama_only':3,'raw_mean':7,'raw_median':7,'sham_median':13,'robust_sham':13,'robust_independent':25}
    for i,op in enumerate(['occlusion','blur','lama']):costs[f'sham_{op}']=5;costs[f'independent_{op}']=9
    costs['robust_sham_occlusion_blur']=9;costs['robust_independent_occlusion_blur']=17
    for model,rows in original['details'].items():
        for task in ['gqa','refcocog']:
            for anchor in ['gold','predicted']:
                selected=[r for r in rows if r['task']==task and r['anchor']==anchor];key=f'{task}/{model}/{anchor}';out[key]={}
                for method,cost in costs.items():
                    arrays=[]
                    for role in ['positive','negative']:
                        values=[]
                        for r in selected:
                            region=r[role]
                            if method in region:value=region[method]
                            elif method=='robust_sham_occlusion_blur':value=robust(region['corrected_effects'][:2])
                            elif method=='robust_independent_occlusion_blur':value=robust(region['independent_effects'][:2])
                            elif method.startswith('sham_'):value=region['corrected_effects'][['occlusion','blur','lama'].index(method[5:])]
                            else:value=region['independent_effects'][['occlusion','blur','lama'].index(method[12:])]
                            values.append(value)
                        arrays.append(values)
                    out[key][method]={'direct_scores_per_image_pair':cost,**paired_summary(*arrays)}
    atomic_json('reports/evidence/v2/E3_inference_budget_frontier.json',{'cells':out,'scope':'Upper-bound unique model-image scores before exact identity/canonical cache reuse, including clean frozen answer anchor. CPU edit costs additional. No post-hoc best-operator claim.'})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,task in zip(axes,['gqa','refcocog']):
        for model,marker in [('qwen','o'),('internvl','s')]:
            for method in ['raw_occlusion','raw_mean','robust_sham','robust_independent']:
                r=out[f'{task}/{model}/gold'][method];ax.scatter(r['direct_scores_per_image_pair'],r['ranking_accuracy'],marker=marker,label=model+'/'+method)
        ax.set_title(task);ax.set_xlabel('Model-image scores / pair');ax.set_ylim(.5,1);ax.set_xticks([3,7,13,25])
    axes[0].set_ylabel('Exploratory oracle pair ranking');axes[1].legend(fontsize=7,loc='lower right');fig.tight_layout();fig.savefig('reports/evidence/v2/E3_budget_frontier.png',dpi=150);plt.close(fig)


if __name__=='__main__':run()
