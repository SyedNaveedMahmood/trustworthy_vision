"""Matched first-five dose diagnostics across all three backbones, no new forwards."""
import json,collections
from pathlib import Path
import numpy as np
from src.scoring import bounded_logit
from src.evaluation.v2 import paired_summary,paired_difference,pair_credit
from src.storage import atomic_json


def run():
    manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items'];result={'status':'EXPLORATORY','cells':{},'selection':'fixed first five initial images/task, shared by all three models','human_audit':'HUMAN_AUDIT_PENDING'}
    lines=['| Task | Model | Factor | Anchor | n | Sham pair ranking | Paired change versus sigma24 / scale1 |','|---|---|---|---|---:|---:|---|']
    for model in ['qwen','smol','internvl']:
        path=Path('reports/evidence/v2/E4_smol_bounded.json' if model=='smol' else f'reports/evidence/v2/E4_{model}.json');record=json.loads(path.read_text())
        if record['status']!='COMPLETE_EXPLORATORY':raise ValueError('Incomplete bounded ablation')
        base=json.loads(Path(f'reports/evidence/v2/E2_pairs_corrected_{model}.json').read_text());baseg=collections.defaultdict(dict)
        for r in base['rows']:baseg[r['task'],r['image_id']][r['variant']]=r
        for task in ['gqa','refcocog']:
            ids=[r['image_id'] for r in manifest if r['dataset']==task][:5]
            for factor in ['blur12','blur36','mask075','mask125']:
                groups=collections.defaultdict(dict)
                for r in record['rows']:
                    if r['task']==task and r['image_id'] in ids and r['factor']==factor:groups[r['image_id']][r['variant']]=r
                if sorted(groups)!=sorted(ids) or any(len(v)!=5 for v in groups.values()):raise ValueError('Matched ablation factorial missing cells')
                for anchor in ['gold','predicted']:
                    values=[];baseline=[]
                    for iid in ids:
                        g=groups[iid];b=baseg[task,iid];idx='ABCD'.index(g['clean']['gold'] if anchor=='gold' else g['clean']['prediction']);op='blur' if factor.startswith('blur') else 'occlusion'
                        logit=lambda row:float(bounded_logit(row['probabilities'][idx])[0])
                        values.append([logit(g[f'r{ri}_sham'])-logit(g[f'r{ri}_removal']) for ri in [0,1]])
                        baseline.append([logit(b[f'r{ri}_{op}_sham'])-logit(b[f'r{ri}_{op}_removal']) for ri in [0,1]])
                    p,n=np.asarray(values).T;stats=paired_summary(p,n);a=[pair_credit(*v) for v in values];b=[pair_credit(*v) for v in baseline];difference=paired_difference(a,b)
                    result['cells'][f'{task}/{model}/{factor}/{anchor}']={**stats,'versus_base_dose':difference,'image_ids':ids}
                    lo,hi=difference['ci95'];lines.append(f"| {task} | {model} | {factor} | {anchor} | 5 | {stats['ranking_accuracy']:.3f} | {difference['effect']:+.3f} [{lo:+.3f},{hi:+.3f}] |")
    atomic_json('reports/evidence/v2/E4_matched_dose.json',result);Path('reports/evidence/v2/E4_matched_dose_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':run()
