"""Retrieve official sources, record immutable revisions and access failures."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
from urllib.request import Request, urlopen

from .storage import atomic_json, digest, retry_network

REPOS = ["huggingface/transformers", "advimman/lama", "lichengunc/refer",
         "tensorflow/datasets", "facebookresearch/segment-anything", "Zhounan1222/CoEV"]
MODELS = {"qwen": "Qwen/Qwen2.5-VL-3B-Instruct", "smol": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}
PAGES = {
    "gqa": "https://cs.stanford.edu/people/dorarad/gqa/download.html",
    "refcocog_download": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    "refcocog_archive": "https://web.archive.org/web/20220413012904id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    "refer_issue": "https://api.github.com/repos/lichengunc/refer/issues/14/comments",
    "evict": "https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Kotte_EVICT_Evidence-Sufficiency_Verification_via_Counterfactual_Dropout_for_Visually-Grounded_Selective_Question_CVPRW_2026_paper.html",
    "coev": "https://papers.miccai.org/miccai-2026/0442-Paper0281.html",
    "csgr": "https://arxiv.org/abs/2609.13228",
    "reality_check": "https://openaccess.thecvf.com/content/CVPR2026W/DataMFM/html/Sar_VLM_Reality_Check_A_Causal_Counterfactual_Benchmark_for_Diagnosing_Cognitive_CVPRW_2026_paper.html",
}


def fetch(url, destination):
    """Keep source snapshots locally; publish only hashes, metadata and URLs."""
    events = []
    def request():
        try:
            with urlopen(Request(url, headers={"User-Agent": "controlled-evidence-reproducibility-audit"}), timeout=20) as r:
                data = r.read()
                events.append({"status": r.status, "bytes": len(data)})
                return data
        except Exception as e:
            events.append({"error": type(e).__name__, "detail": str(e)})
            raise
    record = {"url": url}
    try:
        data, retries = retry_network(request)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        record.update(status="RETRIEVED", sha256=digest(data), bytes=len(data), retries=retries)
    except Exception as e:
        data = None
        record.update(status="UNAVAILABLE", error=type(e).__name__, detail=str(e))
    record["attempts"] = events
    return record, data


def source_audit(root=Path(".")):
    root = Path(root)
    raw = root / "outputs/sources"
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "repositories": {}, "models": {}, "pages": {}}
    for name in REPOS:
        base = f"https://api.github.com/repos/{name}"
        meta, data = fetch(base + "/commits/HEAD", raw / f"{name.replace('/', '_')}_head.json")
        entry = {"head_request": meta}
        if data:
            sha = json.loads(data)["sha"]
            entry["revision"] = sha
            for filename in ("README.md", "LICENSE", "LICENSE.txt"):
                rec, _ = fetch(f"https://raw.githubusercontent.com/{name}/{sha}/{filename}", raw / f"{name.replace('/', '_')}_{filename}")
                entry[filename] = rec
            rec, data = fetch(base + f"/git/trees/{sha}?recursive=1", raw / f"{name.replace('/', '_')}_tree.json")
            entry["tree_request"] = rec
            if data:
                entry["source_files"] = [r["path"] for r in json.loads(data).get("tree", []) if r["path"].endswith(".py")]
        result["repositories"][name] = entry
        atomic_json(root / "reports/evidence/source_inventory.json", result)
    for key, name in MODELS.items():
        rec, data = fetch(f"https://huggingface.co/api/models/{name}", raw / f"{key}_api.json")
        entry = {"request": rec, "model_id": name}
        if data:
            metadata = json.loads(data)
            entry.update(revision=metadata["sha"], gated=metadata.get("gated"), card_data=metadata.get("cardData"))
            for filename in ("README.md", "LICENSE", "config.json", "preprocessor_config.json", "tokenizer_config.json"):
                entry[filename], _ = fetch(f"https://huggingface.co/{name}/resolve/{metadata['sha']}/{filename}", raw / f"{key}_{filename}")
        result["models"][key] = entry
        atomic_json(root / "reports/evidence/source_inventory.json", result)
    for key, url in PAGES.items():
        result["pages"][key], _ = fetch(url, raw / f"{key}.source")
        atomic_json(root / "reports/evidence/source_inventory.json", result)
    return result


def environment(root=Path(".")):
    import psutil
    import torch
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "os": platform.platform(), "ram_total_bytes": psutil.virtual_memory().total,
              "disk_free_bytes": shutil.disk_usage(root).free, "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    result["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], text=True).strip()
    if torch.cuda.is_available():
        result.update(gpu_name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), bf16_supported=torch.cuda.is_bf16_supported())
        torch.cuda.reset_peak_memory_stats()
        start = __import__("time").perf_counter()
        with torch.inference_mode():
            a = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
            b = a @ a
            torch.cuda.synchronize()
            result["bf16_matmul_correct"] = bool((b == 1024).all().item())
        result.update(cuda_probe_seconds=__import__("time").perf_counter() - start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(Path(root) / "reports/evidence/environment.json", result)
    return result
