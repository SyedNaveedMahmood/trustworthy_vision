"""New development replication cohort; same color rule, no model outcome access."""
import collections,json,random,re,zipfile
from pathlib import Path
from scripts.prepare_debug import prepare_ref,get_zip_member
from src.edits import xywh_to_xyxy
from src.storage import atomic_json,digest
from src.data import validate_manifest
from src.session import update

PALETTE=['black','white','red','green','blue','yellow','brown','orange','pink','purple','gray']


def run(purpose='extension'):
    corrected=purpose=='qualification_corrected';partition='qualification_corrected' if corrected else 'development_extension'
    ref_count=20 if corrected else 60;gqa_count=20 if corrected else 40
    ref_seed=20274 if corrected else 20271;gqa_seed=20275 if corrected else 20272
    update('C fixed development extension preparation')
    ref=Path(f'outputs/manifests/refcocog_{partition}.json')
    if not ref.exists() or len(json.loads(ref.read_text()))!=ref_count:prepare_ref(partition,ref_count,ref_seed)
    out=Path(f'outputs/manifests/gqa_{partition}.json')
    existing=[r for p in Path('outputs/manifests').glob('*.json') if p!=out for r in json.loads(p.read_text())]
    existing+=json.loads(Path('outputs/v2/gqa_hard_manifest.json').read_text())
    vg={str(m['image_id']):m.get('coco_id') for m in json.loads(zipfile.ZipFile('outputs/vg_image_data.json.zip').read('image_data.json'))}
    old_gqa={r['image_id'] for r in existing if r['dataset']=='gqa'};old_coco={int(r['image_id']) for r in existing if r['dataset']=='refcocog'}
    questions=json.loads(Path('datasets/gqa/val_balanced_questions.json').read_text());graphs=json.loads(zipfile.ZipFile('outputs/sceneGraphs.zip').read('val_sceneGraphs.json'))
    eligible=[];counts=collections.Counter()
    for qid,q in questions.items():
        iid=str(q['imageId']);reason=None
        if q['groups'].get('global')!='color' or q['types']['structural']!='query' or q['answer'] not in PALETTE:reason='not_closed_basic_color_query'
        elif iid in old_gqa or vg.get(iid) in old_coco:reason='prior_identity'
        else:
            ids=set(re.findall(r'\((\d+)\)',q.get('semanticStr','')));objects=graphs.get(iid,{}).get('objects',{})
            if len(ids)!=1 or not ids.issubset(objects):reason='not_single_reference'
            else:
                obj=objects[next(iter(ids))];g=graphs[iid]
                if q['answer'] not in obj.get('attributes',[]):reason='attribute_not_confirmed'
                elif re.search(r'\b'+re.escape(q['answer'])+r'\b',q['question'].lower()):reason='literal_answer'
                elif min(obj['w'],obj['h'])<24 or not .01<=obj['w']*obj['h']/(g['width']*g['height'])<=.35:reason='target_geometry'
        if reason:counts[reason]+=1
        else:eligible.append((qid,q,next(iter(ids))))
    random.Random(gqa_seed).shuffle(eligible);rows=[];seen=set()
    for qid,q,oid in eligible:
        iid=str(q['imageId'])
        if iid in seen:continue
        seen.add(iid);g=graphs[iid];obj=g['objects'][oid]
        colors=collections.Counter(a for o in g['objects'].values() for a in o.get('attributes',[]) if a in PALETTE and a!=q['answer'])
        distractors=sorted(set(PALETTE)-{q['answer']},key=lambda c:(-colors[c],PALETTE.index(c)))[:3]
        options=[q['answer']]+distractors;random.Random(gqa_seed+int(qid)).shuffle(options)
        image_path=f'gqa/images/{iid}.jpg';get_zip_member('https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip',f'images/{iid}.jpg',Path('datasets')/image_path)
        rows.append({'dataset':'gqa','dataset_version':'questions1.2-sceneGraphs1.1','partition':partition,'experiment':'annotation_defined_basic_colors_'+purpose,
          'image_id':iid,'item_id':qid,'image_path':image_path,'image_sha256':digest((Path('datasets')/image_path).read_bytes()),'question':q['question'],
          'options':options,'gold':'ABCD'[options.index(q['answer'])],'boxes':[xywh_to_xyxy([obj['x'],obj['y'],obj['w'],obj['h']],g['width'],g['height'])],
          'annotation_ids':[oid],'task_subtype':'color_attribute','audit_status':'HUMAN_AUDIT_PENDING','seed':gqa_seed,
          'source_url':'https://cs.stanford.edu/people/dorarad/gqa/download.html','source_license':'GQA CC BY4; original image rights retained'})
        atomic_json(out,rows)
        if len(rows)==gqa_count:break
    # Cross-dataset known COCO mapping: any collisions in proposed Ref source are excluded later, never inferred irrelevant.
    refrows=json.loads(ref.read_text());known_gqa={vg[r['image_id']] for r in existing+rows if r['dataset']=='gqa' and vg.get(r['image_id']) is not None}
    safe_ref=[r for r in refrows if int(r['image_id']) not in known_gqa]
    atomic_json(f'outputs/v2/ref_{purpose}_source.json',safe_ref)
    atomic_json(f'reports/evidence/v2/{purpose}_source_summary.json',{'purpose':purpose,'gqa_count':len(rows),'ref_source_count':len(refrows),'ref_safe_count':len(safe_ref),
        'ref_known_identity_exclusions':len(refrows)-len(safe_ref),'gqa_eligible_questions':len(eligible),'gqa_eligibility_exclusions':dict(counts),
        'gqa_manifest_sha256':digest(rows),'ref_manifest_sha256':digest(safe_ref),'selection_uses_model_outputs':False,
        'items':[{k:r[k] for k in ['dataset','image_id','item_id','image_sha256','seed']} for r in rows+safe_ref]})
    validate_manifest(existing+rows,Path('datasets'))
    update('C fixed development extension preparation',status='COMPLETE',gqa=len(rows),ref_safe=len(safe_ref))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--purpose',choices=['extension','qualification_corrected'],default='extension');run(p.parse_args().purpose)
