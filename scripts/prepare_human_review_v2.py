"""Blinded edit review packets; all human judgment fields deliberately empty."""
import csv,json,random,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from src.storage import atomic_json,digest
from src.session import update
from src.presentation import present
from src.edits.pilot import masks


def run(combined=False):
    cohorts=['corrected','extension'] if combined else ['corrected']
    edits=[{**r,'cohort':cohort} for cohort in cohorts for r in json.loads(Path(f'outputs/v2/edits_pairs_{cohort}.json').read_text())]
    manifest=[r for cohort in cohorts for r in json.loads(Path(f'outputs/v2/pilot_manifest_{cohort}.json').read_text())['items']]
    root=Path('outputs/v2/human_review_combined' if combined else 'outputs/v2/human_review');root.mkdir(parents=True,exist_ok=True)
    if (root/'review_form.csv').exists():
        with (root/'review_form.csv').open(newline='',encoding='utf-8') as stream:
            if any(any(v for k,v in r.items() if k!='review_id') for r in csv.DictReader(stream)):
                raise ValueError('Human ratings present: never overwrite reviewer work')
    groups=collections.defaultdict(list)
    for edit in edits:
        role='candidate' if edit['region_index']<2 else 'independent_control'
        for condition in ['removal','sham']:
            groups[(edit['operator'],role,condition)].append((edit,condition))
    rng=random.Random(20270);chosen=[]
    for key,values in sorted(groups.items()):
        if combined:
            for task in ['gqa','refcocog']:
                subset=[v for v in values if v[0]['task']==task];rng.shuffle(subset);chosen+=subset[:15]
        else:rng.shuffle(values);chosen+=values[:30]
    rng.shuffle(chosen);private=[];form=[];panels=[]
    full=root/'full_resolution';full.mkdir(exist_ok=True)
    for index,(edit,condition) in enumerate(chosen):
        review_id=f'{"V3" if combined else "V2"}_{index+1:04d}';item=next(x for x in manifest if x['dataset']==edit['task'] and x['image_id']==edit['image_id'])
        original=Image.open(item['canonical_path']).convert('RGB');modified=Image.open(edit[condition+'_path']).convert('RGB')
        # Preserve exact scored task overlays before adding the review-only pointer.
        original=present(original,item)[0];modified=present(modified,item)[0]
        original.save(full/f'{review_id}_original.png');modified.save(full/f'{review_id}_edited.png')
        core,extent,_=masks(original.size,edit['box'],edit['params']['halo'])
        core.save(full/f'{review_id}_core_mask.png');extent.save(full/f'{review_id}_expanded_mask.png')
        panel=Image.new('RGB',(700,300),'white');d=ImageDraw.Draw(panel)
        # Neutral pointer reveals intended region geometry, never operator/role/gold.
        for j,im in enumerate([original.copy(),modified.copy()]):
            ImageDraw.Draw(im).rectangle(edit['expanded_box'],outline='#00ffff',width=2)
            thumb=ImageOps.contain(im,(340,250));panel.paste(thumb,(j*350,25))
        d.text((4,4),review_id+'  original | edited; cyan = intended edit extent',fill='black')
        d.text((4,278),item['question'][:100],fill='black')
        panel.save(root/f'{review_id}.jpg',quality=90);panels.append(panel)
        private.append({'review_id':review_id,'task':edit['task'],'image_id':edit['image_id'],'cohort':edit['cohort'],'operator':edit['operator'],'condition':condition,'region_index':edit['region_index'],'edit_cache_key':edit['key'],'region_box':edit['box']})
        form.append({'review_id':review_id,'reviewer_id':'','removal_validity':'','sham_preservation':'','off_target_validity':'','question_referent_region_match':'','referent_fully_covered_by_region':'','unexpected_semantic_changes':'','artifacts':'','confidence_1_to_5':'','notes':''})
    with (root/'review_form.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=form[0]);writer.writeheader();writer.writerows(form)
    atomic_json(root/'PRIVATE_blinding_key.json',private)
    for offset in range(0,len(panels),12):
        sheet=Image.new('RGB',(1400,1800),'#dddddd')
        for j,im in enumerate(panels[offset:offset+12]):sheet.paste(im,((j%2)*700,(j//2)*300))
        sheet.save(root/f'contact_sheet_{offset//12+1:02d}.jpg',quality=90)
    (root/'REVIEW_INSTRUCTIONS.txt').write_text('HUMAN_AUDIT_PENDING. Two independent reviewers required. Do not open PRIVATE_blinding_key.json. Copy review_form.csv separately to review_form_reviewer1.csv and review_form_reviewer2.csv and use distinct real reviewer IDs. Use literal yes/no/uncertain/not_applicable for validity fields and confidence1–5. No judgments have been populated by software. Removal validity: whether the intended original region content/attribute was removed, irrespective of its relevance. Sham preservation: whether original core content is preserved. Off-target validity: whether question-relevant content elsewhere remains intact and no misleading task cue is introduced. Judge source referent matching and complete box coverage separately; a changed patch is not sufficient evidence of successful removal. Inspect full_resolution/{review_id}_original.png and _edited.png at native resolution for every uncertain case. These lossless files include the same white box/letter task overlays seen by the VLM. Cyan extent pointers appear only in thumbnails for review; core/expanded mask PNGs provide exact geometry. Intended operator/role/condition remain hidden; mark not_applicable when a criterion cannot be judged. Independent-control scoring uses the separate key later. Never use these development judgments as confirmatory outcomes.\n',encoding='utf-8')
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
             'private_key_sha256':digest(private),'local_directory':str(root).replace(chr(92),'/'),'geometry':geometry,'cohorts':cohorts,'native_resolution_lossless_pairs':len(form),'scored_task_overlays_preserved':True,
             'limitation':'Automated geometry and exact pixel restoration do not establish semantic intervention validity.'}
    atomic_json('reports/evidence/v2/edit_quality_and_human_review_combined.json' if combined else 'reports/evidence/v2/edit_quality_and_human_review.json',summary)
    update('D blinded human review preparation',status='HUMAN_AUDIT_PENDING',packets=len(form),ratings=0)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--combined',action='store_true');run(p.parse_args().combined)
