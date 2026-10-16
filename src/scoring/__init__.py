"""Preregistered response-sensitivity equations (not causal identification)."""
import numpy as np
from scipy.special import logsumexp


def choice_probabilities(log_scores):
    a = np.asarray(log_scores, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("Expected four finite choice log scores")
    return np.exp(a - logsumexp(a))


def bounded_logit(p, epsilon=1e-6):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability")
    q = np.clip(p, epsilon, 1-epsilon)
    return np.log(q) - np.log1p(-q), int(np.count_nonzero(q != p))


def evidence(clean_p, removed_p, sham_p, control_removed_p, control_sham_p, coefficient=0.5):
    """Controls must come from an independent label-free sampler, not relevance labels."""
    if coefficient not in (0, 0.25, 0.5, 1.0):
        raise ValueError("Coefficient outside preregistered grid")
    clean, n0 = bounded_logit(clean_p)
    removed, n1 = bounded_logit(removed_p)
    sham, n2 = bounded_logit(sham_p)
    crem, n3 = bounded_logit(control_removed_p)
    csham, n4 = bounded_logit(control_sham_p)
    if any(x.shape != (3,) for x in (removed, sham, crem, csham)):
        raise ValueError("Exactly three operators required")
    raw = clean - removed
    d = sham - removed
    z = d - (csham - crem)
    median = np.median(z)
    mad = 1.4826 * np.median(np.abs(z-median))
    return {"raw": raw.tolist(), "sham_corrected": d.tolist(), "z": z.tolist(),
            "median": float(median), "dispersion": float(mad), "score": float(median-coefficient*mad),
            "mean_operator_raw": float(np.mean(raw)), "median_operator_raw": float(np.median(raw)),
            "mean_corrected": float(np.mean(z)), "clamp_count": n0+n1+n2+n3+n4}


def pair_correct(positive, negative):
    a, b = np.asarray(positive), np.asarray(negative)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.shape != b.shape:
        raise ValueError("Invalid paired scores")
    return (a > b).astype(float) + 0.5*(a == b)
