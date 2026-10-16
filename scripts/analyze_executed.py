"""Analyze only actually executed E0/E1 observations; refuse absent primary results."""
import collections
import json
from pathlib import Path

import jsonschema
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.evaluation import paired_cluster_bootstrap
from src.storage import atomic_json,digest

ROOT=Path("reports"); EVIDENCE=ROOT/"evidence"


def wilson(k,n):
    p=k/n; z=norm.ppf(.975); d=1+z*z/n
    center=(p+z*z/(2*n))/d
    margin=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(center-margin),float(center+margin)]


def main():
    schema=json.loads(Path("schemas/score_row.schema.json").read_text())
    models={k:json.loads((EVIDENCE/f"E1_{k}.json").read_text()) for k in ("qwen","smol")}
    summary={"partition":"development","n_per_task":40,"primary_method_effect":None,
             "primary_method_ci":None,"verdict":"INCONCLUSIVE","cells":{},"schema_validated_rows":0}
    bootstrap_rows=[]
    for key,run in models.items():
        rows=run["rows"]
        assert len(rows)==320 and run["status"] in ("COMPETENCE_PASSED","COMPETENCE_FAILED")
        for row in rows:
            jsonschema.validate(row,schema)
            assert abs(sum(row["probabilities"])-1)<1e-12
            assert np.isfinite(row["choice_logits"]).all()
        summary["schema_validated_rows"]+=len(rows)
        for task in ("gqa","refcocog"):
            data=run["metrics"][task]
            cell={"conditions":data,"accuracy_ci95":{c:wilson(data[c]["correct"],40) for c in ("clean","text_only","shuffled_image","choice_rotation")}}
            paired={condition:{r["image_id"]:r for r in rows if r["task"]==task and r["condition"]==condition}
                    for condition in ("clean","text_only","shuffled_image","choice_rotation")}
            ids=sorted(paired["clean"])
            rng=np.random.default_rng(20264)
            sample=rng.integers(0,40,(2000,40))
            gain=np.array([float(paired["clean"][i]["correct"])-float(paired["text_only"][i]["correct"]) for i in ids])
            cell["image_vs_text_paired_bootstrap_ci95"]=np.quantile(gain[sample].mean(axis=1),[.025,.975]).tolist()
            cell["choice_rotation_prediction_disagreement"]=float(np.mean([("ABCD"[("ABCD".index(paired["choice_rotation"][i]["prediction"])-1)%4]!=paired["clean"][i]["prediction"]) for i in ids]))
            clean=[paired["clean"][i] for i in ids]
            confidence=np.array([max(r["probabilities"]) for r in clean]); correct=np.array([r["correct"] for r in clean])
            ece=0.
            for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
                selected=(confidence>=low)&(confidence<high if high<1 else confidence<=high)
                if selected.any(): ece+=float(selected.mean()*abs(correct[selected].mean()-confidence[selected].mean()))
            cell["descriptive_ECE_10_equal_width_bins"]=ece
            cell["subtype_clean_counts"]={subtype:{"n":sum(r["task_subtype"]==subtype for r in clean),
                                                          "correct":sum(r["correct"] for r in clean if r["task_subtype"]==subtype)}
                                                  for subtype in sorted({r["task_subtype"] for r in clean})}
            latencies=[r["wall_seconds"] for r in rows if r["task"]==task and not r["cache_hit"]]
            cell["scoring_latency_seconds"]={"median":float(np.median(latencies)),"p95":float(np.quantile(latencies,.95))}
            summary["cells"][f"{task}/{key}"]=cell
            for i in ids:
                bootstrap_rows.append({"task":task,"model":key,"image_id":i,"candidate_correct":int(paired["clean"][i]["correct"]),
                                       "baseline_correct":int(paired["text_only"][i]["correct"])})
    summary["competence_macro_image_vs_text_contrast_not_H2"]=paired_cluster_bootstrap(bootstrap_rows)
    budget_path=Path("outputs/budget.json") if Path("outputs/budget.json").exists() else EVIDENCE/"budget_snapshot.json"
    budget=json.loads(budget_path.read_text())
    atomic_json(EVIDENCE/"budget_snapshot.json",budget)
    runs=[]
    for name in ("qwen_overlay_v0_smoke.json","smol_overlay_v0_smoke.json","qwen_synthetic_smoke.json","smol_synthetic_smoke.json","smol_processor_failure.json","E1_qwen.json","E1_smol.json"):
        if (EVIDENCE/name).exists():
            run=json.loads((EVIDENCE/name).read_text())
            runs.append({"record":name,"telemetry":run["telemetry"]})
    summary["resources"]={"forward_calls":budget["forward_calls"],"failed_forward_calls":budget["failed_calls"],
                          "synchronized_forward_wall_seconds":budget["forward_wall_seconds"],
                          "synchronized_forward_wall_hours":budget["forward_wall_seconds"]/3600,
                          "instrumented_model_process_wall_seconds":sum(r["telemetry"]["wall_seconds"] for r in runs),
                          "instrumented_model_process_wall_hours":sum(r["telemetry"]["wall_seconds"] for r in runs)/3600,
                          "scope":"Model-process telemetry includes CPU preprocessing/loading; not active GPU kernel hours. LaMa ran on CPU.",
                          "runs":runs}
    atomic_json(EVIDENCE/"executed_summary.json",summary)
    figures=ROOT/"figures"; figures.mkdir(exist_ok=True)
    colors=["#2563eb","#64748b","#d97706","#059669"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,task in zip(axes,("gqa","refcocog")):
        for j,c in enumerate(("clean","text_only","shuffled_image","choice_rotation")):
            values=[]; errors=[]
            for k in ("qwen","smol"):
                cell=summary["cells"][f"{task}/{k}"]; value=cell["conditions"][c]["accuracy"]
                values.append(value); lo,hi=cell["accuracy_ci95"][c]; errors.append([max(0,value-lo),max(0,hi-value)])
            ax.bar(np.arange(2)+(j-1.5)*.18,values,.17,label=c.replace("_"," "),color=colors[j],yerr=np.array(errors).T,capsize=2)
        ax.axhline(.25,color="gray",linestyle=":",linewidth=1,label="25% chance")
        ax.axhline(.4,color="red",linestyle="--",linewidth=1,label="40% feasibility gate")
        ax.set_xticks([0,1],["Qwen 3B","SmolVLM2 2.2B"]); ax.set_title(task.upper()+" — 40 development images")
        ax.set_ylim(0,1.08); ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("Accuracy (Wilson 95% interval)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.suptitle("Competence diagnostics; primary controlled-evidence hypothesis remains untested")
    fig.tight_layout(rect=(0,.15,1,.94))
    fig.savefig(figures/"E1_competence.png",dpi=180); fig.savefig(figures/"E1_competence.svg")
    svg=figures/"E1_competence.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"resources":{k:v for k,v in summary["resources"].items() if k!="runs"},
                      "competence_contrast":summary["competence_macro_image_vs_text_contrast_not_H2"]},indent=2))


if __name__=="__main__": main()
