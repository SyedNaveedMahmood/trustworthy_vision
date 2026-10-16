"""Implemented E0 entrypoints and fail-closed gate reporting.

Later stages deliberately refuse to run until E0 inputs and methodology are valid.
They are not claimed as implemented experiments.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

from .storage import atomic_json, digest


def smoke(model_key):
    import truststore
    truststore.inject_into_ssl()
    from .models.choice import ChoiceModel, Telemetry
    from .synthetic import shapes
    import torch
    source = json.loads(Path("reports/evidence/source_inventory.json").read_text(encoding="utf-8"))["models"][model_key]
    path = Path(f"reports/evidence/{model_key}_synthetic_smoke.json")
    result = {"timestamp_utc":datetime.now(timezone.utc).isoformat(), "dataset":"SYNTHETIC_INFRASTRUCTURE_ONLY",
              "model_revision":source["revision"], "real_dataset_items":0, "rows":[], "status":"RUNNING"}
    atomic_json(path,result)
    telemetry = Telemetry()
    try:
        with telemetry:
            model = ChoiceModel(source["model_id"], source["revision"], model_key)
            for i, fixture in enumerate(shapes()):
                row = model.score(fixture["image"],fixture["question"],fixture["options"],verify=i==0)
                row.update(item_id=fixture["id"], gold=fixture["gold"], synthetic=True)
                result["rows"].append(row)
                atomic_json(path,result)
                print(f"{model_key} {fixture['id']}: {row['prediction']} gold={fixture['gold']}", flush=True)
            result["synthetic_accuracy"] = sum(r["prediction"]==r["gold"] for r in result["rows"])/len(result["rows"])
            from PIL import Image
            from .edits import edit, overlay_boxes
            result["real_rows"] = []
            for task in ("gqa", "refcocog"):
                manifest_path=Path(f"outputs/manifests/{task}_debug.json")
                if not manifest_path.exists():
                    result.setdefault("missing_real_manifests",[]).append(task)
                    continue
                manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
                for i, item in enumerate(manifest):
                    image=Image.open(Path("datasets")/item["image_path"]).convert("RGB")
                    variants=[("clean",image)]
                    if i<4:
                        variants.append(("occlusion",edit(image,item["boxes"]["ABCD".index(item["gold"]) if task=="refcocog" else 0],"occlusion")))
                    for condition, variant in variants:
                        if task=="refcocog": variant=overlay_boxes(variant,item["boxes"])
                        row=model.score(variant,item["question"],item["options"],verify=i==0 and condition=="clean")
                        row.update(task=task,item_id=item["item_id"],image_id=item["image_id"],
                                   source_image_sha256=item["image_sha256"],gold=item["gold"],condition=condition,synthetic=False)
                        result["real_rows"].append(row)
                        print(f"{model_key} {task} {i+1}/12 {condition}: {row['prediction']} gold={item['gold']}",flush=True)
                        atomic_json(path,result)
            result["real_dataset_items"]=len({(r["task"],r["image_id"]) for r in result["real_rows"]})
            result["status"] = "COMPUTATIONAL_SMOKE_COMPLETE_NOT_FULL_E0_PASS"
    except Exception as e:
        result.update(status="FAILED", exception=type(e).__name__, detail=str(e))
        Path("outputs").mkdir(exist_ok=True)
        Path(f"outputs/{model_key}_smoke_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        result["telemetry"] = telemetry.result
        if Path("outputs/budget.json").exists():
            result["cumulative_budget"] = json.loads(Path("outputs/budget.json").read_text())
        atomic_json(path,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit-sources","environment","synthetic-smoke","prepare-data","validate-images","run","lock","report"])
    parser.add_argument("--config", default="configs/pilot_v1.yaml")
    parser.add_argument("--model", choices=["qwen","smol"])
    parser.add_argument("--split", default="debug", choices=["debug","development","confirmatory"])
    parser.add_argument("--stage", choices=[f"E{i}" for i in range(7)])
    args = parser.parse_args()
    import yaml
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pins=yaml.safe_load(Path("configs/runtime_pins.yaml").read_text(encoding="utf-8"))
    if pins.get("protocol_config_sha256") and digest(config)!=pins["protocol_config_sha256"]:
        parser.error("This implementation was run only with the hashed pilot_v1 protocol. Record and implement amendments before using changed config.")
    if args.command == "audit-sources":
        from .audit import source_audit
        source_audit()
    elif args.command == "environment":
        from .audit import environment
        print(json.dumps(environment(),indent=2))
    elif args.command == "synthetic-smoke":
        if not args.model:
            parser.error("--model required")
        smoke(args.model)
    elif args.command == "prepare-data" and args.split in ("debug","development"):
        import subprocess
        return subprocess.run([sys.executable,"-m","scripts.prepare_debug" if args.split=="debug" else "scripts.prepare_development"]).returncode
    elif args.command == "validate-images":
        from .data import validate_manifest
        rows=[]
        for task in ("gqa","refcocog"):
            p=Path(f"outputs/manifests/{task}_{args.split}.json")
            if not p.exists():
                print(f"BLOCKED: no {task} {args.split} manifest")
                return 2
            rows+=json.loads(p.read_text(encoding="utf-8"))
        print(json.dumps(validate_manifest(rows,Path("datasets")),indent=2))
    elif args.command == "prepare-data":
        print("BLOCKED: confirmatory sampling requires passed prerequisites and a genuine protocol lock")
        return 2
    elif args.command == "run" and args.stage == "E1":
        import subprocess
        for model in ([args.model] if args.model else ["qwen","smol"]):
            result=subprocess.run([sys.executable,"-m","scripts.run_competence","--model",model])
            if result.returncode: return result.returncode
    elif args.command in ("run","lock"):
        print("BLOCKED: the two-model competence prerequisite failed (SmolVLM2 RefCOCOg 15/40 < 40%). "
              "Fallback requires manual access. Operator human audit pending; control design unresolved. "
              "Later experiment orchestration is unfinished; see reports/STATUS.md.")
        return 2
    elif args.command == "report":
        print(Path("reports/STATUS.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
