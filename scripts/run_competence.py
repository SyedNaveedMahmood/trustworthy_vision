"""E1 development feasibility screen; no confirmatory inference or method tuning.

E0 flagged SmolVLM's grounding competence. This diagnostic quantifies that flag
on the prespecified 40 new images/task, without declaring E0 passed.
"""
import argparse
import collections
import json
from pathlib import Path
import time

import truststore
truststore.inject_into_ssl()
import numpy as np
from PIL import Image

from src.models.choice import ChoiceModel, Telemetry
from src.edits import overlay_boxes
from src.storage import atomic_json,digest
from src.data import validate_manifest


def run(model_key):
    source=json.loads(Path("reports/evidence/source_inventory.json").read_text())["models"][model_key]
    out=Path(f"reports/evidence/E1_{model_key}.json")
    record={"stage":"E1","model":model_key,"status":"RUNNING","partition":"development",
            "E0_qualification":"COMPUTATIONAL_SMOKE_COMPLETE; grounding competence flagged, not full E0 pass",
            "rows":[],"confirmatory_items":0}
    atomic_json(out,record)
    telemetry=Telemetry()
    try:
        with telemetry:
            model=ChoiceModel(source["model_id"],source["revision"],model_key)
            for task in ("gqa","refcocog"):
                items=json.loads(Path(f"outputs/manifests/{task}_development.json").read_text())
                assert len(items)==40
                validate_manifest(items,Path("datasets"))
                for i,item in enumerate(items):
                    image=Image.open(Path("datasets")/item["image_path"]).convert("RGB")
                    other=Image.open(Path("datasets")/items[(i+1)%len(items)]["image_path"]).convert("RGB").resize(image.size)
                    if task=="refcocog":
                        image=overlay_boxes(image,item["boxes"])
                        other=overlay_boxes(other,item["boxes"])
                    options=item["options"]
                    conditions=[("clean",image,options,item["gold"]),
                                ("text_only",None,options,item["gold"]),
                                ("shuffled_image",other,options,item["gold"]),
                                ("choice_rotation",image,options[-1:]+options[:-1],"ABCD"[("ABCD".index(item["gold"])+1)%4])]
                    for condition,variant,choices,gold in conditions:
                        row=model.score(variant,item["question"],choices)
                        row.update(task=task,item_id=item["item_id"],image_id=item["image_id"],condition=condition,
                                   gold=gold,correct=row["prediction"]==gold,source_image_sha256=item["image_sha256"],
                                   task_subtype=item["task_subtype"],partition="development",
                                   target_box_area=None if task=="gqa" else (item["boxes"]["ABCD".index(item["gold"])][2]-item["boxes"]["ABCD".index(item["gold"])][0])*(item["boxes"]["ABCD".index(item["gold"])][3]-item["boxes"]["ABCD".index(item["gold"])][1])/(image.width*image.height))
                        record["rows"].append(row)
                        atomic_json(out,record)
                    print(model_key,task,f"{i+1}/40",flush=True)
            record["metrics"]={}
            for task in ("gqa","refcocog"):
                metrics={}
                for condition in ("clean","text_only","shuffled_image","choice_rotation"):
                    rows=[r for r in record["rows"] if r["task"]==task and r["condition"]==condition]
                    p=np.array([r["probabilities"] for r in rows]); gold=np.array(["ABCD".index(r["gold"]) for r in rows])
                    metrics[condition]={"n":len(rows),"correct":sum(r["correct"] for r in rows),
                                        "accuracy":float(np.mean([r["correct"] for r in rows])),
                                        "nll":float(-np.log(p[np.arange(len(rows)),gold]).mean()),
                                        "entropy":float(-(p*np.log(p)).sum(axis=1).mean()),
                                        "brier":float(((p-np.eye(4)[gold])**2).sum(axis=1).mean()),
                                        "prediction_counts":dict(collections.Counter(r["prediction"] for r in rows))}
                metrics["image_over_text_accuracy_gain"]=metrics["clean"]["accuracy"]-metrics["text_only"]["accuracy"]
                metrics["competence_gate"]=metrics["clean"]["accuracy"]>=.4 and metrics["image_over_text_accuracy_gain"]>0
                record["metrics"][task]=metrics
            record["status"]="COMPETENCE_PASSED" if all(v["competence_gate"] for v in record["metrics"].values()) else "COMPETENCE_FAILED"
    except Exception as e:
        record.update(status="ERROR",exception=type(e).__name__,detail=str(e))
        raise
    finally:
        record["telemetry"]=telemetry.result
        record["cumulative_budget"]=model.budget.value if "model" in locals() else None
        atomic_json(out,record)
    print(record["status"],json.dumps(record["metrics"],indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True,choices=["qwen","smol"])
    run(p.parse_args().model)
