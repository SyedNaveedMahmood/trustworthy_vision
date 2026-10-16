"""Validation for real manifest preparation; never substitute synthetic records."""
from pathlib import Path
from ..storage import digest


def validate_manifest(rows, dataset_root=None):
    identities, hashes = {}, {}
    for r in rows:
        if len(r["options"]) != 4 or len(set(r["options"])) != 4 or r["gold"] not in "ABCD" or len(r["gold"]) != 1:
            raise ValueError("Invalid four-choice mapping")
        if r["partition"] not in ("debug", "development", "qualification", "development_hard", "confirmatory"):
            raise ValueError("Unknown partition")
        key = (r["dataset"], str(r["image_id"]))
        if key in identities:
            raise ValueError("Repeated image ID across or within partitions")
        if r["image_sha256"] in hashes:
            raise ValueError("Duplicate image content across datasets/partitions")
        identities[key] = r["partition"]
        hashes[r["image_sha256"]] = r["partition"]
        p = Path(r["image_path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Public manifests need relative dataset paths")
        if dataset_root is not None and digest((Path(dataset_root)/p).read_bytes()) != r["image_sha256"]:
            raise ValueError("Image content hash mismatch")
        for key in ("source_url", "source_license", "audit_status", "annotation_ids", "task_subtype", "seed"):
            if key not in r:
                raise ValueError(f"Missing manifest field: {key}")
        if r["audit_status"] not in ("NOT_REVIEWED","HUMAN_AUDIT_PENDING","HUMAN_VALIDATED","INVALID"):
            raise ValueError("Unknown audit status")
        if r["audit_status"] == "HUMAN_VALIDATED" and len(set(r.get("human_rater_ids",[]))) < 2:
            raise ValueError("Two independent human rater records required")
    return {"count": len(rows), "sha256": digest(rows)}


def input_inventory(root=Path("datasets")):
    required = {"gqa": ["gqa/questions.json", "gqa/scene_graphs.json", "gqa/images"],
                "refcocog": ["refcocog/refs(umd).p", "refcocog/instances.json", "coco/train2014"]}
    return {name: {"required": files, "missing": [p for p in files if not (Path(root)/p).exists()]}
            for name,files in required.items()}
