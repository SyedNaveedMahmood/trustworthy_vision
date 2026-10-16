"""Image-paired exploratory inference. Ties receive half credit."""
import numpy as np
from scipy.stats import rankdata


def pair_credit(positive,negative):
    return float(positive>negative)+.5*float(positive==negative)


def auc(positive,negative):
    p=np.asarray(positive,dtype=float);n=np.asarray(negative,dtype=float)
    if not len(p) or not len(n) or not np.isfinite(np.r_[p,n]).all(): raise ValueError('Invalid AUROC inputs')
    ranks=rankdata(np.r_[p,n])
    return float((ranks[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))


def paired_summary(positive,negative,replicates=2000,seed=20269):
    p=np.asarray(positive,dtype=float);n=np.asarray(negative,dtype=float)
    if p.ndim!=1 or p.shape!=n.shape or not len(p): raise ValueError('Complete image pairs required')
    credit=(p>n).astype(float)+.5*(p==n)
    indices=np.random.default_rng(seed).integers(0,len(p),(replicates,len(p)))
    means=credit[indices].mean(axis=1)
    bp=p[indices];bn=n[indices]
    aucs=((bp[:,:,None]>bn[:,None,:])+.5*(bp[:,:,None]==bn[:,None,:])).mean(axis=(1,2))
    # Wilson bound is supplementary only for genuinely binary, tie-free pair outcomes.
    center=(credit.mean()+1.96**2/(2*len(p)))/(1+1.96**2/len(p))
    half=1.96*np.sqrt(credit.mean()*(1-credit.mean())/len(p)+1.96**2/(4*len(p)**2))/(1+1.96**2/len(p))
    return {'n':len(p),'ranking_accuracy':float(credit.mean()),'ranking_ci95':np.quantile(means,[.025,.975]).tolist(),
            'tie_count':int(np.sum(credit==.5)),'supplementary_wilson_ci95_tie_free':[max(0,float(center-half)),min(1,float(center+half))] if not np.any(credit==.5) else None,
            'pairwise_auroc':float(credit.mean()),'pooled_region_auroc':auc(p,n),
            'pooled_region_auroc_ci95':np.quantile(aucs,[.025,.975]).tolist(),
            'negative_score_gt_1_rate':float(np.mean(n>1)),
            'positive_mean':float(p.mean()),'negative_mean':float(n.mean()),
            'bootstrap_unit':'image pairs; same indices for every method/model','seed':seed,'replicates':replicates}


def paired_difference(a,b,replicates=5000,seed=20269):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a): raise ValueError('Complete aligned image scores required')
    d=a-b;indices=np.random.default_rng(seed).integers(0,len(d),(replicates,len(d)))
    return {'effect':float(d.mean()),'ci95':np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist(),'n':len(d),'seed':seed}
