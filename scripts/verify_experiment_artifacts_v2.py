"""CPU verification of real score factorials, caches, normalization and controls."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from src.evaluation.integrity import validate_scoring_inputs
from src.storage import atomic_json,read_checkpoint,digest
from src.scoring import choice_probabilities
from src.edits.pilot import independent_control
from src.session import update


def run(require_complete=False):
    result={'status':'RUNNING','stages':{},'errors':[]};count=0
    for cohort in ['corrected','extension']:
        manifest=json.loads(Path(f'outputs/v2/pilot_manifest_{cohort}.json').read_text())['items'];edits=json.loads(Path(f'outputs/v2/edits_pairs_{cohort}.json').read_text())
        integrity=validate_scoring_inputs(manifest,edits);lookup={(r['dataset'],r['image_id']):r for r in manifest}
        for item in manifest:
            size=Image.open(item['canonical_path']).size
            for ri in [0,1]:
                expected=list(independent_control(size,item['regions'][ri],20266+int(item['image_id'])))
                if expected!=item['regions'][ri+2]:raise ValueError('Independent control reconstruction mismatch')
        for model in ['qwen','smol','internvl']:
            path=Path(f'reports/evidence/v2/E2_pairs_{cohort}_{model}.json')
            if not path.exists():
                if require_complete:raise ValueError('Missing required stage '+str(path))
                continue
            record=json.loads(path.read_text())
            if record['status']!='COMPLETE_COMPUTATIONAL_ONLY':
                if require_complete:raise ValueError('Incomplete required stage '+str(path))
                continue
            if len(record['rows'])!=integrity['expected_score_rows']:raise ValueError('Wrong score count')
            cells=set();clamps=0;cache_hits=0;mass=[]
            for row in record['rows']:
                identity=(row['task'],row['image_id']);item=lookup[identity];cell=identity+(row['variant'],)
                if cell in cells:raise ValueError('Duplicate score cell')
                cells.add(cell)
                if row['canonical_image_sha256']!=item['canonical_sha256'] or row['gold']!=item['gold']:raise ValueError('Canonical/gold mapping mismatch')
                probs=np.asarray(row['probabilities'])
                if len(set(row['label_ids']))!=4 or len(row['label_ids'])!=4:raise ValueError('Invalid choice-label IDs')
                if not np.allclose(probs,choice_probabilities(row['choice_logits']),atol=1e-12,rtol=0):raise ValueError('Stored probability normalization mismatch')
                if row['prediction']!='ABCD'[int(probs.argmax())] or row['correct']!=(row['prediction']==row['gold']):raise ValueError('Answer mapping mismatch')
                if row['processor_config']['long_edge']!=768:raise ValueError('Mixed input caps')
                if model=='smol' and row['processor_config'].get('label_continuation_separator')!=' ':raise ValueError('Bare-token Smol entered corrected E2')
                payload=read_checkpoint(Path('outputs/cache')/f"{row['run_key']}.json",row['run_key'])
                if payload['probabilities']!=row['probabilities'] or payload['image_sha256']!=row['image_sha256']:raise ValueError('Stage/cache payload disagreement')
                clamps+=int(np.sum((probs<1e-6)|(probs>1-1e-6)));cache_hits+=int(row['cache_hit'])
                if 'answer_step_total_label_mass' in row:mass.append(row['answer_step_total_label_mass'])
            result['stages'][cohort+'/'+model]={'score_rows':len(record['rows']),'cache_hits_during_original_stage':cache_hits,
                'probability_clamps_if_all4_answers_scored':clamps,'all4_probability_values':4*len(record['rows']),
                'canonical_label_mass_coverage_rows':len(mass),'min_total_canonical_label_mass':min(mass) if mass else None,
                'input_integrity':integrity,'complete_cache_checksums_verified':True,'control_geometry_reconstructed':True}
            count+=len(record['rows'])
    result.update(status='PASS_CPU_ARTIFACT_VERIFICATION',verified_score_rows=count,scope='Computational integrity only; no semantic or human validity pass')
    atomic_json('reports/evidence/v2/artifact_verification.json',result);update('G experiment artifact verification',status='COMPLETE',verified_rows=count)
    print(count,'score rows verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');run(p.parse_args().require_complete)
