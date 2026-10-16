"""Fixed descriptive geometry/confidence/language strata; no sample deletion."""
import json,re
from pathlib import Path
import numpy as np
from src.evaluation.v2 import pair_credit,paired_difference
from src.storage import atomic_json


def run():
    result=json.loads(Path('reports/evidence/v2/E3_comparison.json').read_text());manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items'];out={}
    for model,rows in result['details'].items():
        for task in ['gqa','refcocog']:
            for anchor in ['gold','predicted']:
                items=[r for r in rows if r['task']==task and r['anchor']==anchor]
                strata={}
                for row in items:
                    item=next(x for x in manifest if x['dataset']==task and x['image_id']==row['image_id'])
                    from PIL import Image
                    from src.edits.pilot import area,overlap
                    im=Image.open(item['canonical_path']);fraction=area(item['regions'][0])/(im.width*im.height)
                    labels=['all','original_correct' if row['original_correct'] else 'original_incorrect','confidence_ge_0.8' if row['original_confidence']>=.8 else 'confidence_lt_0.8',
                            'target_area_lt_0.05' if fraction<.05 else 'target_area_ge_0.05','area_mismatch_le_0.15' if item['area_mismatch']<=.15 else 'area_mismatch_gt_0.15',
                            'control_overlaps_opposite_reference' if max(item['controls_overlap_opposite_reference'])>0 else 'control_no_opposite_overlap']
                    if task=='refcocog':
                        gi='ABCD'.index(item['gold']);maxiou=max(overlap(item['boxes'][gi],b) for j,b in enumerate(item['boxes']) if j!=gi)
                        labels+=['target_distractor_iou_gt_0.1' if maxiou>.1 else 'target_distractor_iou_le_0.1',
                                 'relation_keyword' if re.search(r'\b(left|right|behind|between|under|holding|next to|in front)\b',item['question'],re.I) else 'no_relation_keyword']
                    for label in labels:strata.setdefault(label,[]).append(row)
                for label,subset in strata.items():
                    strata[label]={'n':len(subset),'ranking':{method:float(np.mean([pair_credit(r['positive'][method],r['negative'][method]) for r in subset])) for method in ['raw_occlusion','raw_mean','sham_median','robust_sham','robust_independent']}}
                out[f'{task}/{model}/{anchor}']=strata
    atomic_json('reports/evidence/v2/E4_stratified_diagnostics.json',{'status':'EXPLORATORY','strata':out,'note':'Fixed cutoffs and keyword heuristic describe observed failures; relation keywords are not verified relational-reasoning labels. Small overlapping strata are not independent hypothesis tests.'})


if __name__=='__main__':run()
