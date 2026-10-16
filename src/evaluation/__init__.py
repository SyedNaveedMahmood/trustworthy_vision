"""Paired inference with shared-image clusters across the two model families."""
import numpy as np


def paired_cluster_bootstrap(rows, replicates=2000, seed=20264):
    """Rows: task, model, image_id, candidate_correct, baseline_correct.

    Requires complete shared images for both models; no silent missing-cell deletion.
    Resamples images ONCE per task then carries both models together.
    """
    if not rows:
        raise ValueError("No experimental observations: cannot estimate a CI")
    tasks = sorted({r["task"] for r in rows})
    models = sorted({r["model"] for r in rows})
    if len(tasks) != 2 or len(models) != 2:
        raise ValueError("Primary estimand requires two tasks and two models")
    lookup = {}
    for r in rows:
        k = (r["task"], r["model"], r["image_id"])
        if k in lookup:
            raise ValueError("Duplicate image/model cell")
        if r["candidate_correct"] not in (0, 0.5, 1) or r["baseline_correct"] not in (0, 0.5, 1):
            raise ValueError("Pair-ranking indicators must be 0, 0.5 or 1")
        lookup[k] = r["candidate_correct"]-r["baseline_correct"]
    matrices = []
    for t in tasks:
        ids = [sorted({r["image_id"] for r in rows if r["task"] == t and r["model"] == m}) for m in models]
        if not ids[0] or ids[0] != ids[1]:
            raise ValueError("Missing shared-image cell; explicit missingness analysis required")
        matrices.append(np.array([[lookup[t,m,i] for m in models] for i in ids[0]]))
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates)
    for matrix in matrices:
        indices = rng.integers(0, len(matrix), (replicates, len(matrix)))
        draws += matrix[indices].mean(axis=(1,2)) / len(tasks)
    return {"effect": float(np.mean([x.mean() for x in matrices])),
            "ci95": np.quantile(draws, [0.025,0.975]).tolist(), "replicates": replicates,
            "seed": seed, "unit": "image_id clustered jointly across models within task"}
