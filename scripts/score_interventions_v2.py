"""Resumable E2 computational scores on fixed images, with all answer probabilities."""
import argparse,json
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from PIL import Image
from src.presentation import present
from src.storage import atomic_json,digest
from src.session import update,check_window


def run(key,mode='pairs',cohort='corrected'):
    manifest_path=Path(f'outputs/v2/pilot_manifest_{cohort}.json') if cohort!='legacy' else Path('outputs/v2/pilot_manifest.json')
    manifest=json.loads(manifest_path.read_text())
    edits_path=Path(f'outputs/v2/edits_{mode}_{cohort}.json')
    if cohort=='legacy' and not edits_path.exists(): edits_path=Path(f'outputs/v2/edits_{mode}.json')
    edits=json.loads(edits_path.read_text()); mh=digest(manifest['items'])
    from src.evaluation.integrity import validate_scoring_inputs
    integrity=validate_scoring_inputs(manifest['items'],edits,mode)
    out=Path(f'reports/evidence/v2/E2_{mode}_{cohort}_{key}.json')
    result=json.loads(out.read_text()) if out.exists() else {'stage':'E2_EXPLORATORY','cohort':cohort,'mode':mode,'model':key,'manifest_sha256':mh,'status':'RUNNING','human_audit':'HUMAN_AUDIT_PENDING','rows':[]}
    if result['manifest_sha256']!=mh: raise ValueError('Pilot manifest changed; no checkpoint mixing')
    if result.get('edit_manifest_sha256',integrity['edit_manifest_sha256'])!=integrity['edit_manifest_sha256']:raise ValueError('Edited image manifest changed; no checkpoint mixing')
    result['edit_manifest_sha256']=integrity['edit_manifest_sha256'];result['status']='RUNNING'
    done={(r['task'],r['image_id'],r['variant']) for r in result['rows']}
    selected={(e['task'],e['image_id']) for e in edits}
    expected={(t,i,'clean') for t,i in selected}|{(e['task'],e['image_id'],f"r{e['region_index']}_{e['operator']}_{c}") for e in edits for c in ['removal','sham']}
    if len(done)!=len(result['rows']) or not done<=expected:raise ValueError('Duplicate or unknown score checkpoint cells')
    if done==expected:
        print('Complete input-hash-verified checkpoint: zero model loads and zero new forwards',flush=True)
        return
    from src.models.choice import ChoiceModel,Telemetry
    tm=Telemetry();update('D E2 computational scoring',model=key,mode=mode,cohort=cohort,completed=len(done))
    try:
        with tm:
            if key=='internvl':
                from src.models.internvl import InternChoiceModel
                model=InternChoiceModel(long_edge=768)
            else:
                pins=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
                model=ChoiceModel(pins['model_id'],pins['revision'],key,long_edge=768)
            for item in manifest['items']:
                task=item['dataset'];iid=item['image_id']
                if (task,iid) not in selected: continue
                variants=[('clean',item['canonical_path'],None)]
                for edit in edits:
                    if edit['task']==task and edit['image_id']==iid:
                        for condition in ['removal','sham']:
                            variants.append((f"r{edit['region_index']}_{edit['operator']}_{condition}",edit[condition+'_path'],edit))
                for variant,path,edit in variants:
                    check_window()
                    if (task,iid,variant) in done: continue
                    image=Image.open(path).convert('RGB')
                    image,q,options,gold=present(image,item)
                    row=model.score(image,q,options)
                    row.update(task=task,image_id=iid,item_id=item['item_id'],variant=variant,gold=gold,correct=row['prediction']==gold,
                               edit_params=edit['params'] if edit else None,edit_cache_key=edit['key'] if edit else None,
                               canonical_image_sha256=item['canonical_sha256'],cohort=cohort,mode=mode)
                    result['rows'].append(row);done.add((task,iid,variant));atomic_json(out,result)
                update('D E2 computational scoring',model=key,mode=mode,completed=len(done),last_image_id=iid)
            if len(result['rows'])!=integrity['expected_score_rows']:raise ValueError('Incomplete score factorial')
            result['status']='COMPLETE_COMPUTATIONAL_ONLY'
    except Exception as e:
        result.update(status='ERROR',error=type(e).__name__,detail=str(e));raise
    finally:
        if hasattr(tm,'result'): result['telemetry']=tm.result
        result['cumulative_budget']=model.budget.value if 'model' in locals() else None
        atomic_json(out,result)
    update('D E2 computational scoring',status='COMPLETE_COMPUTATIONAL_ONLY',model=key,mode=mode,score_rows=len(result['rows']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['qwen','smol','internvl']);p.add_argument('--mode',choices=['pairs','grid'],default='pairs');p.add_argument('--cohort',choices=['corrected','legacy','extension'],default='corrected');a=p.parse_args();run(a.model,a.mode,a.cohort)
