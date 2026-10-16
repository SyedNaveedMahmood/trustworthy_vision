"""Blinded edit review packets; all human judgment fields deliberately empty."""
import csv,json,random,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from src.storage import atomic_json,digest
from src.session import update


def run():
    edits=json.loads(Path('outputs/v2/edits_pairs_corrected.json').read_text())
    manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items']
    root=Path('outputs/v2/human_review');root.mkdir(parents=True,exist_ok=True)
    groups=collections.defaultdict(list)
    for edit in edits:
        role='candidate' if edit['region_index']<2 else 'independent_control'
        for condition in ['removal','sham']:
            groups[(edit['operator'],role,condition)].append((edit,condition))
    rng=random.Random(20270);chosen=[]
    for key,values in sorted(groups.items()):
        rng.shuffle(values);chosen+=values[:30]
    rng.shuffle(chosen);private=[];form=[];panels=[]
    for index,(edit,condition) in enumerate(chosen):
        review_id=f'V2_{index+1:04d}';item=next(x for x in manifest if x['dataset']==edit['task'] and x['image_id']==edit['image_id'])
        original=Image.open(item['canonical_path']).convert('RGB');modified=Image.open(edit[condition+'_path']).convert('RGB')
        panel=Image.new('RGB',(700,300),'white');d=ImageDraw.Draw(panel)
        for j,im in enumerate([original,modified]):
            thumb=ImageOps.contain(im,(340,250));panel.paste(thumb,(j*350,25))
        d.text((4,4),review_id+'  original | edited',fill='black')
        d.text((4,278),item['question'][:100],fill='black')
        panel.save(root/f'{review_id}.jpg',quality=90);panels.append(panel)
        private.append({'review_id':review_id,'task':edit['task'],'image_id':edit['image_id'],'operator':edit['operator'],'condition':condition,'region_index':edit['region_index'],'edit_cache_key':edit['key'],'region_box':edit['box']})
        form.append({'review_id':review_id,'reviewer_id':'','removal_validity':'','sham_preservation':'','off_target_validity':'','unexpected_semantic_changes':'','artifacts':'','confidence_1_to_5':'','notes':''})
    with (root/'review_form.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=form[0]);writer.writeheader();writer.writerows(form)
    atomic_json(root/'PRIVATE_blinding_key.json',private)
    for offset in range(0,len(panels),12):
        sheet=Image.new('RGB',(1400,1800),'#dddddd')
        for j,im in enumerate(panels[offset:offset+12]):sheet.paste(im,((j%2)*700,(j//2)*300))
        sheet.save(root/f'contact_sheet_{offset//12+1:02d}.jpg',quality=90)
    (root/'REVIEW_INSTRUCTIONS.txt').write_text('HUMAN_AUDIT_PENDING. Two independent reviewers required. Do not open PRIVATE_blinding_key.json. Compare original and edit against the referring question. Use yes/no/uncertain/not-applicable for validity, preservation, unintended semantics and artifacts; confidence 1–5. Unknown intended condition is allowed: judge visible content, do not infer successful removal merely from a changed patch. No judgments have been populated by the software. Independent-control validity requires later keyed analysis; preserve blindness during individual ratings. Never use these development judgments as confirmatory outcomes.\n',encoding='utf-8')
    geometry={'edit_requests':len(edits),'rendered_images':2*len(edits),'by_operator':{}}
    for op in ['occlusion','blur','lama']:
        rows=[e for e in edits if e['operator']==op]
        geometry['by_operator'][op]={'n':len(rows),'outside_mask_max_change':max(e['outside_mask_max_change'] for e in rows),
                                  'sham_core_max_change':max(e['sham_core_max_change'] for e in rows),
                                  'mean_core_fraction':float(np.mean([e['core_fraction'] for e in rows])),
                                  'mean_expanded_fraction':float(np.mean([e['mask_fraction'] for e in rows])),
                                  'mean_inside_absolute_change':float(np.mean([e['mean_abs_change_inside'] for e in rows])),
                                  'identity_removals':sum(e['removal_identical'] for e in rows),'identity_shams':sum(e['sham_identical'] for e in rows),
                                  'raw_lama_outside_max_change':max((e.get('raw_lama_outside_max_change') or 0) for e in rows)}
    summary={'status':'HUMAN_AUDIT_PENDING','completed_ratings':0,'independent_reviewers':0,'seed':20270,'review_packet_count':len(form),
             'per_operator_role_condition':{'/'.join(k):min(30,len(v)) for k,v in groups.items()},
             'private_key_sha256':digest(private),'local_directory':'outputs/v2/human_review','geometry':geometry,
             'limitation':'Automated geometry and exact pixel restoration do not establish semantic intervention validity.'}
    atomic_json('reports/evidence/v2/edit_quality_and_human_review.json',summary)
    update('D blinded human review preparation',status='HUMAN_AUDIT_PENDING',packets=len(form),ratings=0)


if __name__=='__main__':run()
