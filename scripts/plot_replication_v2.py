"""Standalone figures for actual paired development comparisons."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation.v2 import pair_credit,paired_difference
from src.evaluation import paired_cluster_bootstrap
from src.storage import atomic_json


def run():
    record=json.loads(Path('reports/evidence/v2/E3_replication.json').read_text());data=record['details']['pooled'];best=record['comparator_proposal']['method'];primary=['qwen','smol'];methods=['raw_occlusion','raw_mean','raw_median','sham_median','robust_sham','robust_independent'];summary={}
    for method in methods:
        rows=[{'task':r['task'],'model':m,'image_id':r['image_id'],'candidate_correct':pair_credit(r['positive'][method],r['negative'][method]),'baseline_correct':0} for m in primary for r in data[m] if r['anchor']=='gold']
        summary[method]=paired_cluster_bootstrap(rows,5000,20269)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(methods));means=np.array([summary[m]['effect'] for m in methods]);bounds=np.array([summary[m]['ci95'] for m in methods])
    ax.errorbar(x,means,yerr=np.array([means-bounds[:,0],bounds[:,1]-means]),fmt='o',capsize=4)
    ax.set_xticks(x,methods,rotation=20,ha='right');ax.set_ylim(.5,1);ax.set_ylabel('Macro pair-ranking accuracy');ax.set_title('Qwen + corrected Smol; 40 color GQA / 41 RefCOCOg images\nExploratory image-cluster95% intervals; semantic audit pending');fig.tight_layout();fig.savefig('reports/evidence/v2/E3_pooled_primary.png',dpi=180);plt.close(fig)
    effects=[];labels=[]
    for task in ['gqa','refcocog']:
        for model in primary:
            rows=[r for r in data[model] if r['task']==task and r['anchor']=='gold'];a=[pair_credit(r['positive']['robust_independent'],r['negative']['robust_independent']) for r in rows];b=[pair_credit(r['positive'][best],r['negative'][best]) for r in rows]
            effects.append(paired_difference(a,b));labels.append(f'{task} / {model} (n={len(rows)})')
    effects.append(record['comparator_proposal']['candidate_vs_selected_baseline']);labels.append('Equal-weight task/model macro')
    fig,ax=plt.subplots(figsize=(9,4));y=np.arange(len(effects));point=np.array([r['effect'] for r in effects]);bounds=np.array([r['ci95'] for r in effects]);ax.errorbar(point,y,xerr=np.array([point-bounds[:,0],bounds[:,1]-point]),fmt='o',capsize=4)
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Robust-independent minus raw-median pair ranking');ax.set_title('Descriptive paired95% intervals; baseline selected on this development data');fig.tight_layout();fig.savefig('reports/evidence/v2/E3_pooled_contrasts.png',dpi=180);plt.close(fig)
    cost={'raw_occlusion':3,'raw_mean':7,'raw_median':7,'sham_median':13,'robust_sham':13,'robust_independent':25}
    fig,ax=plt.subplots(figsize=(8,4))
    for i,m in enumerate(methods):
        x=cost[m];y=summary[m]['effect'];ax.scatter(x,y,s=45);ax.annotate(m,(x,y),xytext=(4,4 if i%2 else -12),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Direct model-image scores per pair');ax.set_ylabel('Macro oracle-region ranking');ax.set_ylim(.75,.92);ax.set_xticks([3,7,13,25]);ax.set_title('Exploratory accuracy/cost; additional CPU edits not included');fig.tight_layout();fig.savefig('reports/evidence/v2/E3_cost_accuracy.png',dpi=180);plt.close(fig)
    atomic_json('reports/evidence/v2/E3_pooled_macro_summary.json',{'methods':summary,'contrast_vs_selected_baseline':record['comparator_proposal']['candidate_vs_selected_baseline'],'baseline':best,'status':'EXPLORATORY_NOT_CONFIRMATORY'})


if __name__=='__main__':run()
