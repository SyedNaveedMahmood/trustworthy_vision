"""Annotation/overlay/split audits, with no automatic semantic/human validity claims."""
import collections,json,pickle,re,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from scipy.fft import dctn
from src.storage import atomic_json,digest
from src.edits import overlay_boxes
from src.edits.pilot import overlap,area
from src.session import update


def perceptual_hash(image):
    gray=np.asarray(image.convert('L').resize((32,32)),dtype=float)
    low=dctn(gray,norm='ortho')[:8,:8].flatten(); bits=low>np.median(low[1:])
    return sum(int(v)<<i for i,v in enumerate(bits))


def main():
    update('C dataset, distractor and split audit')
    qdata=json.loads(Path('datasets/gqa/val_balanced_questions.json').read_text())
    gqa=json.loads(Path('outputs/manifests/gqa_development.json').read_text())
    ga={'n':len(gqa),'answer_group_counts':dict(collections.Counter(str(qdata[r['item_id']]['groups'].get('global')) for r in gqa)),
        'answer_letter_counts':dict(collections.Counter(r['gold'] for r in gqa)),'items':[]}
    for row in gqa:
        q=qdata[row['item_id']]; gold=row['options']['ABCD'.index(row['gold'])]
        ga['items'].append({'image_id':row['image_id'],'item_id':row['item_id'],'global_group':q['groups'].get('global'),
                             'structural_type':q['types']['structural'],'mixed_null_answer_pool':q['groups'].get('global') is None,
                             'binary_reference_in_four_choices':gold in ['yes','no'],
                             'literal_answer_in_question':bool(re.search(r'\b'+re.escape(gold)+r'\b',row['question'].lower()))})
    for key in ['mixed_null_answer_pool','binary_reference_in_four_choices','literal_answer_in_question']: ga[key+'_count']=sum(r[key] for r in ga['items'])
    refdata=json.loads(Path('datasets/refcocog/instances.json').read_text()); refs=pickle.loads(Path('datasets/refcocog/refs(umd).p').read_bytes(),encoding='latin1')
    refmap={str(r['ref_id']):r for r in refs}; exprmap=collections.defaultdict(set)
    for r in refs:
        for sent in r['sentences']: exprmap[(r['image_id'],sent['sent'].strip().lower())].add(r['ann_id'])
    refscores=[]
    for item in json.loads(Path('outputs/manifests/refcocog_development.json').read_text()):
        ref=refmap[item['item_id']]; goldidx='ABCD'.index(item['gold']); positive=item['boxes'][goldidx]
        im=Image.open(Path('datasets')/item['image_path']).convert('RGB'); over=overlay_boxes(im,item['boxes'])
        changed=np.any(np.asarray(im)!=np.asarray(over),axis=2); x0,y0,x1,y1=[int(x) for x in positive]
        target=changed[y0:y1,x0:x1]
        refscores.append({'image_id':item['image_id'],'mapping_correct':str(ref['image_id'])==item['image_id'] and str(ref['ann_id'])==item['annotation_ids'][goldidx],
                          'duplicate_expression_other_referent':len(exprmap[(ref['image_id'],ref['sentences'][0]['sent'].strip().lower())])>1,
                          'target_fraction':area(positive)/(im.width*im.height),
                          'max_target_distractor_iou':max(overlap(positive,b) for j,b in enumerate(item['boxes']) if j!=goldidx),
                          'target_pixels_changed_by_overlay_fraction':float(target.mean()) if target.size else None})
    rows=[r for p in Path('outputs/manifests').glob('*.json') for r in json.loads(p.read_text())]
    hard=Path('outputs/v2/gqa_hard_manifest.json')
    if hard.exists(): rows+=json.loads(hard.read_text())
    vg={str(v['image_id']):v.get('coco_id') for v in json.loads(zipfile.ZipFile('outputs/vg_image_data.json.zip').read('image_data.json'))}
    indexed=[]; ids={}; bytehash={}; coco={}; duplicates=[]
    for row in rows:
        key=(row['dataset'],row['image_id']); c=vg.get(row['image_id']) if row['dataset']=='gqa' else int(row['image_id'])
        for kind,k,seen in [('id',key,ids),('byte',row['image_sha256'],bytehash),('coco',c,coco)]:
            if k is not None and k in seen: duplicates.append({'type':kind,'first':seen[k],'second':[row['dataset'],row['image_id'],row['partition']]})
            if k is not None: seen[k]=[row['dataset'],row['image_id'],row['partition']]
        image=Image.open(Path('datasets')/row['image_path']).convert('RGB')
        indexed.append({'dataset':row['dataset'],'image_id':row['image_id'],'partition':row['partition'],'phash':perceptual_hash(image),'pixel_sha256':digest(image.tobytes()+str(image.size).encode()),'has_coco_mapping':c is not None})
    near=[]
    for i,a in enumerate(indexed):
        for b in indexed[:i]:
            dist=(a['phash']^b['phash']).bit_count()
            if dist<=4: near.append({'first':{k:a[k] for k in ['dataset','image_id','partition']},'second':{k:b[k] for k in ['dataset','image_id','partition']},'phash_distance':dist,'exact_pixels':a['pixel_sha256']==b['pixel_sha256']})
    result={'gqa_legacy':ga,'refcocog':{'n':len(refscores),'items':refscores},
            'split_integrity':{'n':len(indexed),'hard_duplicates':duplicates,'perceptual_near_duplicates':near,'gqa_null_coco_mappings':sum(r['dataset']=='gqa' and not r['has_coco_mapping'] for r in indexed),
                               'phash_threshold':4,'interpretation':'Near matches require inspection; null identities and pHash cannot prove complete photographic disjointness.'}}
    atomic_json('reports/evidence/v2/dataset_audit.json',result)
    # Correct/incorrect panels are diagnostic examples, never independent human ratings.
    directory=Path('outputs/v2/diagnostic_panels'); directory.mkdir(parents=True,exist_ok=True)
    items=json.loads(Path('outputs/manifests/refcocog_development.json').read_text())
    for model in ['qwen','smol']:
        results=json.loads(Path(f'reports/evidence/E1_{model}.json').read_text())['rows']
        scored={r['image_id']:r for r in results if r['task']=='refcocog' and r['condition']=='clean'}
        panel=Image.new('RGB',(1200,1000),'#eeeeee'); draw=ImageDraw.Draw(panel)
        for i,item in enumerate(items[:12]):
            image=overlay_boxes(Image.open(Path('datasets')/item['image_path']).convert('RGB'),item['boxes'])
            image.thumbnail((390,220)); x=i%3*400;y=i//3*250;panel.paste(image,(x,y))
            r=scored[item['image_id']];draw.text((x,y+222),f"{item['image_id']} pred={r['prediction']} gold={r['gold']} correct={r['correct']}",fill='black')
        panel.save(directory/f'{model}_first12_outcome_unselected.jpg')
    update('C dataset, distractor and split audit',status='COMPLETE',legacy_null_group=ga['mixed_null_answer_pool_count'],hard_duplicate_count=len(duplicates),near_duplicate_count=len(near))

if __name__=='__main__': main()
