"""Risk-coverage diagnostics; score direction fixed before analysis."""
import numpy as np


def risk_coverage(correct,scores):
    correct=np.asarray(correct,dtype=float);scores=np.asarray(scores,dtype=float)
    if correct.ndim!=1 or correct.shape!=scores.shape or not len(correct) or not np.isfinite(scores).all():raise ValueError('Invalid selective prediction arrays')
    if not np.isin(correct,[0,1]).all():raise ValueError('Correctness must be binary')
    order=np.argsort(-scores,kind='stable');risks=(1-correct[order]).cumsum()/np.arange(1,len(correct)+1)
    oracle=(1-correct[np.argsort(-correct,kind='stable')]).cumsum()/np.arange(1,len(correct)+1)
    return {'aurc':float(risks.mean()),'excess_aurc':float(risks.mean()-oracle.mean()),'risk_curve':risks.tolist(),
            'risk_at_coverage':{str(c):float(risks[max(0,int(np.ceil(c*len(correct)))-1)]) for c in [.5,.7,.85,.95]},
            'tie_count':len(scores)-len(set(scores.tolist())),'tie_rule':'stable fixed image order; sensitivity to ties reported separately'}


def paired_aurc_bootstrap(rows,replicates=2000,seed=20276):
    tasks=sorted({r['task'] for r in rows});models=sorted({r['model'] for r in rows});matrices=[]
    if len(tasks)!=2 or len(models)!=2:raise ValueError('Two tasks/two shared-image backbones required')
    lookup={(r['task'],r['model'],r['image_id']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate selective cell')
    for task in tasks:
        ids=[sorted({r['image_id'] for r in rows if r['task']==task and r['model']==m}) for m in models]
        if not ids[0] or ids[0]!=ids[1]:raise ValueError('Missing shared images')
        matrices.append(np.array([[[lookup[task,m,i][k] for k in ['correct','candidate_score','baseline_score']] for m in models] for i in ids[0]],dtype=float))
    def contrast(matrix):return np.mean([risk_coverage(matrix[:,j,0],matrix[:,j,1])['aurc']-risk_coverage(matrix[:,j,0],matrix[:,j,2])['aurc'] for j in range(len(models))])
    point=float(np.mean([contrast(a) for a in matrices]));rng=np.random.default_rng(seed);draws=[]
    for _ in range(replicates):draws.append(np.mean([contrast(a[rng.integers(0,len(a),len(a))]) for a in matrices]))
    return {'candidate_minus_confidence_aurc':point,'ci95':np.quantile(draws,[.025,.975]).tolist(),'lower_is_better':True,'replicates':replicates,'seed':seed,'unit':'same image indices across models, stratified by task'}
