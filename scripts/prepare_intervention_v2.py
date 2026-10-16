"""Select region pairs without model outputs and cache full three-operator edits."""
import argparse,json,os,random,subprocess,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from src.edits import xywh_to_xyxy,resize_box
from src.edits.pilot import area,overlap,independent_control,grid_candidates,request
from src.storage import atomic_json,digest
from src.session import update,check_window

OUT=Path('outputs/v2/pilot_manifest.json')


def select(cohort='legacy'):
    questions=json.loads(Path('datasets/gqa/val_balanced_questions.json').read_text())
    graphs=json.loads(zipfile.ZipFile('outputs/sceneGraphs.zip').read('val_sceneGraphs.json'))
    items=[]; excluded=[]
    for task in ['gqa','refcocog']:
        if cohort=='extension':source_path=Path('outputs/manifests/gqa_development_extension.json') if task=='gqa' else Path('outputs/v2/ref_extension_source.json')
        else:source_path=Path('outputs/v2/gqa_hard_manifest.json') if task=='gqa' and cohort=='corrected' else Path(f'outputs/manifests/{task}_development.json')
        source=json.loads(source_path.read_text()); random.Random(20266).shuffle(source)
        chosen=0
        for item in source:
            if chosen>=(25 if cohort=='extension' else 16): break
            image=Image.open(Path('datasets')/item['image_path']).convert('RGB'); oldsize=image.size
            image.thumbnail((768,768)); scale=lambda b:resize_box(b,oldsize,image.size)
            positive=item['boxes']['ABCD'.index(item['gold'])] if task=='refcocog' else item['boxes'][0]
            positive=scale(positive)
            if task=='refcocog': negatives=[scale(b) for j,b in enumerate(item['boxes']) if j!='ABCD'.index(item['gold'])]
            else:
                graph=graphs[item['image_id']]; q=questions[item['item_id']]
                # All explicit program IDs are protected from negative selection.
                import re
                protected=set(item['annotation_ids'])|set(re.findall(r'\((\d+)\)',q.get('semanticStr','')))
                negatives=[]
                for oid,obj in graph['objects'].items():
                    if oid in protected: continue
                    try: negatives.append(scale(xywh_to_xyxy([obj['x'],obj['y'],obj['w'],obj['h']],*oldsize)))
                    except ValueError: pass
            reason=None
            if not (.01<=area(positive)/(image.width*image.height)<=.35): reason='target_area_outside_1_to_35pct'
            negatives=[b for b in negatives if min(b[2]-b[0],b[3]-b[1])>=16 and overlap(positive,b)<=.1 and abs(area(b)/area(positive)-1)<=.5]
            if not negatives: reason=reason or 'no_nonoverlapping_negative_within_50pct_area'
            if reason:
                excluded.append({'task':task,'image_id':item['image_id'],'reason':reason}); continue
            negative=min(negatives,key=lambda b:(abs(area(b)/area(positive)-1),sum(abs(x-y) for x,y in zip(b,positive))))
            try:
                controls=[independent_control(image.size,b,20266+int(item['image_id'])) for b in [positive,negative]]
            except ValueError as e:
                excluded.append({'task':task,'image_id':item['image_id'],'reason':str(e)}); continue
            canonical=Path('outputs/v2/canonical')/f"{task}_{item['image_id']}.png"; canonical.parent.mkdir(parents=True,exist_ok=True); image.save(canonical)
            region_boxes=[positive,negative]+controls
            row={**item,'canonical_path':str(canonical).replace(chr(92),'/'),'canonical_sha256':digest(canonical.read_bytes()),
                 'boxes':[scale(b) for b in item['boxes']], 'regions':[list(b) for b in region_boxes],
                 'region_order':['r_positive','r_negative','control_0','control_1'],
                 'area_mismatch':abs(area(negative)/area(positive)-1),'negative_semantics':'annotation_reference_not_human_validated',
                 'control_selection_inputs':['image_size','candidate_coordinates','fixed_seed'],
                 'controls_overlap_opposite_reference':[overlap(controls[0],negative),overlap(controls[1],positive)]}
            items.append(row); chosen+=1
    atomic_json(OUT,{'items':items,'excluded':excluded,'status':'SELECTED_BEFORE_E2_MODEL_OUTCOMES','seed':20266})
    atomic_json(f'reports/evidence/v2/pilot_sample_summary_{cohort}.json',{'cohort':cohort,'task_counts':{t:sum(r['dataset']==t for r in items) for t in ['gqa','refcocog']},
                'items':[{k:r[k] for k in ['dataset','image_id','item_id','canonical_sha256','area_mismatch','controls_overlap_opposite_reference']} for r in items],
                'exclusions':excluded,'manifest_sha256':digest(items),'human_audit':'HUMAN_AUDIT_PENDING'})
    return json.loads(OUT.read_text())


def prepare(mode='pairs',cohort='legacy'):
    global OUT
    OUT=Path(f'outputs/v2/pilot_manifest_{cohort}.json') if cohort!='legacy' else OUT
    manifest=json.loads(OUT.read_text()) if OUT.exists() else select(cohort)
    update('D E2 edit preparation',mode=mode,n=len(manifest['items']))
    entries=[]
    for index,item in enumerate(manifest['items']):
        image=Image.open(item['canonical_path']).convert('RGB')
        if mode=='grid':
            # First eight preselected IDs per task; geometry uses no annotations.
            preceding=sum(r['dataset']==item['dataset'] for r in manifest['items'][:index])
            if preceding>=(12 if cohort=='extension' else 8): continue
            boxes=grid_candidates(image.size)
            regions=boxes+[independent_control(image.size,b,20267+int(item['image_id'])) for b in boxes]
        else: regions=item['regions']
        for rindex,box in enumerate(regions):
            for operator in ['occlusion','blur','lama']:
                check_window(); edit=request(image,box,operator)
                entries.append({'task':item['dataset'],'image_id':item['image_id'],'region_index':rindex,'operator':operator,**edit})
    pending=[e for e in entries if e['operator']=='lama' and not Path(e['removal_path']).exists()]
    if pending:
        root=Path.cwd(); repo=root/'third_party/repos/lama'; env=os.environ.copy()
        env.update(PYTHONPATH=str(repo),TORCH_HOME=str(root/'third_party_weights'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        # Official predictor walks all input masks. Completed raw outputs are moved to
        # a separate batch input directory via links/copies so restart avoids repeats.
        batch=root/f'outputs/v2/lama/batch_{mode}_{cohort}_{digest(sorted(e["key"] for e in pending))[:12]}'; batch.mkdir(parents=True,exist_ok=True)
        import shutil
        for e in pending:
            for suffix in ['.png','_mask001.png']:
                shutil.copy2(root/'outputs/v2/lama/input'/f"{e['key']}{suffix}",batch/f"{e['key']}{suffix}")
        log=root/f'outputs/v2/lama/{mode}_{cohort}_official.log'; (root/'outputs/v2/lama/output').mkdir(parents=True,exist_ok=True)
        command=[str(root/'.venv-lama/Scripts/python.exe'),str(repo/'bin/predict.py'),f"model.path={(root/'third_party_weights/big-lama').as_posix()}",f'indir={batch.as_posix()}',f"outdir={(root/'outputs/v2/lama/output').as_posix()}"]
        start=time.perf_counter(); update('D official LaMa CPU batch',mode=mode,pending=len(pending))
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(f'reports/evidence/v2/lama_batch_{mode}_{cohort}.json',{'returncode':result.returncode,'pending_requests':len(pending),'wall_seconds':time.perf_counter()-start,'device':'cpu','source_revision':'786f5936b27fb3dacd2b1ad799e4de968ea697e7'})
        if result.returncode: raise RuntimeError('Official LaMa failed; see local batch log. No substitute.')
        # Finalize compositing and geometric measurements only after real inference.
        for e in entries:
            if e['operator']=='lama':
                item=next(i for i in manifest['items'] if i['dataset']==e['task'] and i['image_id']==e['image_id'])
                e.update(request(Image.open(item['canonical_path']),e['box'],'lama'))
    atomic_json(f'outputs/v2/edits_{mode}_{cohort}.json',entries)
    update('D E2 edit preparation',status='COMPLETE_COMPUTATIONAL_ONLY',mode=mode,edit_requests=len(entries))
    return entries

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['pairs','grid'],default='pairs'); p.add_argument('--cohort',choices=['legacy','corrected','extension'],default='corrected'); args=p.parse_args(); prepare(args.mode,args.cohort)
