"""Download only official debug inputs using HTTP byte ranges; never touch confirmatory data.

Run with python -m scripts.prepare_debug. Pickle comes exclusively from the CRC-checked
archived official RefCOCOg archive. Never load a user-supplied untrusted pickle.
"""
import collections
import json
from pathlib import Path
import pickle
import random
import zipfile

import truststore
truststore.inject_into_ssl()
import requests
from remotezip import RemoteZip

from src.storage import digest, atomic_json
from src.edits import xywh_to_xyxy
from src.data import validate_manifest

ROOT = Path("datasets")


def get_zip_member(url, name, path):
    path=Path(path)
    if path.exists(): return path.read_bytes()
    with RemoteZip(url, timeout=30) as z:
        data=z.read(name)  # zipfile verifies decompressed size and member CRC
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    return data


def get_image(url,path):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        r=requests.get(url,timeout=30); r.raise_for_status()
        path.write_bytes(r.content)
    from PIL import Image
    with Image.open(path) as image: image.verify()
    return digest(path.read_bytes())


def prepare_ref():
    data=json.loads((ROOT/"refcocog/instances.json").read_text())
    refs=pickle.loads((ROOT/"refcocog/refs(umd).p").read_bytes(),encoding="latin1")
    by_image=collections.defaultdict(list)
    images={r["id"]:r for r in data["images"]}
    annotations={r["id"]:r for r in data["annotations"]}
    licenses={r["id"]:r for r in data["licenses"]}
    for a in data["annotations"]:
        if a.get("iscrowd",0)==0 and a["bbox"][2]>=24 and a["bbox"][3]>=24:
            by_image[a["image_id"]].append(a)
    eligible=[r for r in refs if r["split"]=="train" and len(by_image[r["image_id"]])>=4 and r["ann_id"] in annotations]
    random.Random(20261).shuffle(eligible)
    rows=[]; seen=set()
    for ref in eligible:
        iid=ref["image_id"]
        if iid in seen: continue
        seen.add(iid)
        info=images[iid]; target=annotations[ref["ann_id"]]
        candidates=[a for a in by_image[iid] if a["id"]!=target["id"]]
        if len(candidates)<3: continue
        candidates.sort(key=lambda a:(a["category_id"]!=target["category_id"],abs(a["area"]-target["area"]),a["id"]))
        candidates=[target]+candidates[:3]
        random.Random(20261+iid).shuffle(candidates)
        gold="ABCD"[[a["id"] for a in candidates].index(target["id"])]
        image_path=f"coco/train2014/{info['file_name']}"
        # Official COCO distribution URL is HTTP. Its HTTPS virtual host currently
        # presents a mismatched certificate; do not disable TLS verification.
        url=f"http://images.cocodataset.org/train2014/{info['file_name']}"
        sha=get_image(url,ROOT/image_path)
        rows.append({"dataset":"refcocog","dataset_version":"archived-official-UMD","image_id":str(iid),"image_sha256":sha,
                     "image_path":image_path,"item_id":str(ref["ref_id"]),"partition":"debug","seed":20261,
                     "question":"Select the box matching this description: "+ref["sentences"][0]["sent"],
                     "options":["box A","box B","box C","box D"],"gold":gold,
                     "boxes":[xywh_to_xyxy(a["bbox"],info["width"],info["height"]) for a in candidates],
                     "annotation_ids":[str(a["id"]) for a in candidates],"task_subtype":"referring_expression",
                     "source_url":url,"source_license":licenses.get(info["license"],{}),"audit_status":"HUMAN_AUDIT_PENDING",
                     "diagnostic_only":True})
        print("refcocog debug",len(rows),iid,flush=True)
        atomic_json("outputs/manifests/refcocog_debug.json",rows)
        if len(rows)==12: break
    if len(rows)!=12: raise RuntimeError("Insufficient debug inputs")
    return rows


def prepare_gqa():
    questions=json.loads(get_zip_member("https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
                                      "val_balanced_questions.json",ROOT/"gqa/val_balanced_questions.json"))
    with zipfile.ZipFile("outputs/sceneGraphs.zip") as z:
        print("scene graph members",z.namelist(),flush=True)
        name=next(n for n in z.namelist() if n.endswith("val_sceneGraphs.json"))
        graphs=json.loads(z.read(name))
    pools=collections.defaultdict(set)
    for q in questions.values():
        pools[q["groups"]["global"]].add(q["answer"])
    eligible=[]
    for qid,q in questions.items():
        if len(pools[q["groups"]["global"]])<4: continue
        iid=str(q["imageId"])
        objects=graphs.get(iid,{}).get("objects",{})
        ids=set(str(v) for k in ("question","answer") for v in q.get("annotations",{}).get(k,{}).values())
        if len(ids)!=1 or not ids.issubset(objects): continue
        obj=objects[next(iter(ids))]
        if obj["w"]<24 or obj["h"]<24: continue
        eligible.append((qid,q,ids))
    random.Random(20261).shuffle(eligible)
    rows=[]; seen=set()
    for qid,q,ids in eligible:
        iid=str(q["imageId"])
        if iid in seen: continue
        seen.add(iid)
        distractors=sorted(pools[q["groups"]["global"]]-{q["answer"]})
        rng=random.Random(20261+int(qid)); options=[q["answer"]]+rng.sample(distractors,3); rng.shuffle(options)
        image_path=f"gqa/images/{iid}.jpg"
        get_zip_member("https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",f"images/{iid}.jpg",ROOT/image_path)
        obj=graphs[iid]["objects"][next(iter(ids))]
        box=xywh_to_xyxy([obj["x"],obj["y"],obj["w"],obj["h"]],graphs[iid]["width"],graphs[iid]["height"])
        rows.append({"dataset":"gqa","dataset_version":"questions1.2-sceneGraphs1.1","image_id":iid,
                     "image_sha256":digest((ROOT/image_path).read_bytes()),"image_path":image_path,"item_id":qid,
                     "partition":"debug","seed":20261,"question":q["question"],"options":options,
                     "gold":"ABCD"[options.index(q["answer"])],"boxes":[box],"annotation_ids":sorted(ids),
                     "task_subtype":q["types"]["semantic"],"source_url":"https://cs.stanford.edu/people/dorarad/gqa/download.html",
                     "source_license":"Official GQA site CC BY 4.0; underlying image rights retained by original owners",
                     "audit_status":"HUMAN_AUDIT_PENDING","diagnostic_only":True})
        print("gqa debug",len(rows),iid,flush=True)
        atomic_json("outputs/manifests/gqa_debug.json",rows)
        if len(rows)==12: break
    if len(rows)!=12: raise RuntimeError("Insufficient debug inputs")
    return rows


if __name__=="__main__":
    results=[]
    for name,fn in (("refcocog",prepare_ref),("gqa",prepare_gqa)):
        try:
            rows=fn(); result=validate_manifest(rows,ROOT)
            atomic_json(f"reports/evidence/{name}_debug_manifest_summary.json",
                        {**result,"status":"INPUTS_PREPARED_NOT_E0_PASS","items":[{k:r[k] for k in ("dataset","image_id","item_id","image_sha256")} for r in rows]})
            results+=rows
        except Exception as e:
            atomic_json(f"reports/evidence/{name}_debug_data_failure.json",{"exception":type(e).__name__,"detail":str(e)})
            raise
    print(validate_manifest(results,ROOT))
