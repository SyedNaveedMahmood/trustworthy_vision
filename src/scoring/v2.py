"""Exploratory amended estimators with independent, candidate-specific controls."""
import numpy as np
from . import bounded_logit


def mad(values):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all(): raise ValueError("Non-finite operator effects")
    return float(1.4826*np.median(abs(a-np.median(a))))


def robust(values, coefficient=.5):
    a=np.asarray(values,dtype=float)
    return float(np.median(a)-coefficient*mad(a))


def estimators(clean, removals, shams, control_removals, control_shams, coefficient=.5):
    arrays=[np.asarray(x,dtype=float) for x in (removals,shams,control_removals,control_shams)]
    if any(a.shape!=(3,) for a in arrays): raise ValueError("Three operators required")
    clean_l,n=bounded_logit(clean)
    ls=[]
    for a in arrays:
        l,k=bounded_logit(a); ls.append(l); n+=k
    rem,sham,crem,csham=ls
    raw=clean_l-rem; d=sham-rem; control=csham-crem; z=d-control
    out={"raw_occlusion":float(raw[0]),"blur_only":float(raw[1]),"lama_only":float(raw[2]),
         "raw_mean":float(raw.mean()),"raw_median":float(np.median(raw)),
         "sham_mean":float(d.mean()),"sham_median":float(np.median(d)),
         "robust_sham":robust(d,coefficient),"independent_control_mean":float(z.mean()),
         "independent_control_median":float(np.median(z)),"robust_independent":robust(z,coefficient),
         "offtarget_without_sham":float(np.median(raw-(clean_l-crem))),
         "raw_effects":raw.tolist(),"sham_effects":(clean_l-sham).tolist(),
         "corrected_effects":d.tolist(),"control_effects":control.tolist(),"independent_effects":z.tolist(),
         "dispersion_sham":mad(d),"dispersion_independent":mad(z),"clamp_count":n}
    for i in range(3):
        out[f"loo_{i}"]=robust(np.delete(d,i),coefficient)
    for c in (0,.25,.5,1): out[f"lambda_{c}"]=robust(d,c)
    return out
