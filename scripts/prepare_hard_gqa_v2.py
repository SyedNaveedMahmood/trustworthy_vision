"""Separate color-focused diagnostic using annotation-only eligibility, not VLM errors."""
import collections,json,random,re,zipfile
from pathlib import Path
from scripts.prepare_debug import get_zip_member
from src.edits import xywh_to_xyxy
from src.storage import atomic_json,digest
from src.session import update

PALETTE=['black','white','red','green','blue','yellow','brown','orange','pink','purple','gray']
out=Path('outputs/v2/gqa_hard_manifest.json')
if out.exists():
    print('Preserving existing harder-subset manifest'); raise SystemExit(0)
questions=json.loads(Path('datasets/gqa/val_balanced_questions.json').read_text())
graphs=json.loads(zipfile.ZipFile('outputs/sceneGraphs.zip').read('val_sceneGraphs.json'))
existing=[r for p in Path('outputs/manifests').glob('*.json') for r in json.loads(p.read_text())]
excluded_ids={r['image_id'] for r in existing if r['dataset']=='gqa'}
vg={str(m['image_id']):m.get('coco_id') for m in json.loads(zipfile.ZipFile('outputs/vg_image_data.json.zip').read('image_data.json'))}
coco_seen={int(r['image_id']) for r in existing if r['dataset']=='refcocog'}
eligible=[]; exclusion_counts=collections.Counter()
for qid,q in questions.items():
    reason=None; iid=str(q['imageId'])
    if q['groups'].get('global')!='color' or q['types']['structural']!='query' or q['answer'] not in PALETTE: reason='not_closed_basic_color_query'
    elif iid in excluded_ids or vg.get(iid) in coco_seen: reason='prior_partition_identity'
    else:
        ids=set(re.findall(r'\((\d+)\)',q.get('semanticStr','')))
        objects=graphs.get(iid,{}).get('objects',{})
        if len(ids)!=1 or not ids.issubset(objects): reason='not_single_explicit_program_reference'
        else:
            obj=objects[next(iter(ids))]; graph=graphs[iid]
            if q['answer'] not in obj.get('attributes',[]): reason='answer_not_confirmed_by_scene_attribute'
            elif re.search(r'\b'+re.escape(q['answer'])+r'\b',q['question'].lower()): reason='literal_answer_in_question'
            elif min(obj['w'],obj['h'])<24 or not .01<=obj['w']*obj['h']/(graph['width']*graph['height'])<=.35: reason='target_visibility_or_area'
    if reason: exclusion_counts[reason]+=1
    else: eligible.append((qid,q,next(iter(ids))))
random.Random(20268).shuffle(eligible)
rows=[]; seen=set()
update('C annotation-defined harder GQA data',eligible_questions=len(eligible))
for qid,q,oid in eligible:
    iid=str(q['imageId'])
    if iid in seen: continue
    seen.add(iid); graph=graphs[iid]; obj=graph['objects'][oid]
    colors=collections.Counter(a for o in graph['objects'].values() for a in o.get('attributes',[]) if a in PALETTE and a!=q['answer'])
    candidates=sorted(set(PALETTE)-{q['answer']},key=lambda c:(-colors[c],PALETTE.index(c)))
    options=[q['answer']]+candidates[:3]; random.Random(20268+int(qid)).shuffle(options)
    path=f'gqa/images/{iid}.jpg'
    get_zip_member('https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip',f'images/{iid}.jpg',Path('datasets')/path)
    row={'dataset':'gqa','dataset_version':'questions1.2-sceneGraphs1.1','partition':'development_hard','experiment':'annotation_defined_basic_colors',
         'image_id':iid,'item_id':qid,'image_path':path,'image_sha256':digest((Path('datasets')/path).read_bytes()),
         'question':q['question'],'options':options,'gold':'ABCD'[options.index(q['answer'])],
         'boxes':[xywh_to_xyxy([obj['x'],obj['y'],obj['w'],obj['h']],graph['width'],graph['height'])],
         'annotation_ids':[oid],'task_subtype':'color_attribute','audit_status':'HUMAN_AUDIT_PENDING','seed':20268,
         'source_url':'https://cs.stanford.edu/people/dorarad/gqa/download.html','source_license':'GQA site CC BY 4.0; original photograph rights retained'}
    rows.append(row); atomic_json(out,rows); print('hard GQA',len(rows),flush=True)
    if len(rows)==20: break
atomic_json('reports/evidence/v2/hard_gqa_manifest_summary.json',{'count':len(rows),'eligible_questions':len(eligible),'eligible_unique_images':len({str(q['imageId']) for _,q,_ in eligible}),
            'exclusion_counts':dict(exclusion_counts),'rule':'single reference, basic color query, answer in scene attributes, no literal answer leak, plausible same-type image-present-preferred distractors',
            'manifest_sha256':digest(rows),'items':[{k:r[k] for k in ['image_id','item_id','image_sha256','seed']} for r in rows]})
update('C annotation-defined harder GQA data',status='COMPLETE',count=len(rows))
