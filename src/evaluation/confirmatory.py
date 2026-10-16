"""Fail-closed confirmatory preparation; never infer validity from absent human records."""
from dataclasses import dataclass,asdict
import random
from ..storage import digest
from ..data import validate_manifest


@dataclass(frozen=True)
class Readiness:
    amended_protocol_locked:bool=False
    task_scope_qualified:bool=False
    independent_controls_validated:bool=False
    operator_human_audit_passed:bool=False
    two_independent_raters:bool=False
    frozen_single_comparator:bool=False
    image_disjointness_verified:bool=False


def blockers(readiness,current_calls,samples_per_task=120,models=2,tasks=2,calls_per_item=25,cap=12000):
    reasons=[name for name,value in asdict(readiness).items() if value is not True]
    projected=samples_per_task*models*tasks*calls_per_item
    if current_calls+projected>cap:reasons.append('cumulative_forward_budget_infeasible')
    return {'ready':not reasons,'blockers':reasons,'current_calls':current_calls,'projected_additional_calls':projected,
            'projected_total_calls':current_calls+projected,'cap':cap,
            'maximum_per_task_with_current_factorial':max(0,(cap-current_calls)//(models*tasks*calls_per_item)),
            'note':'Maximum is a projection, never authorization to change the frozen sample plan.'}


def generate_confirmatory_manifest(pool,development,readiness,current_calls,per_task=120,seed=20273):
    gate=blockers(readiness,current_calls,per_task)
    if not gate['ready']:raise RuntimeError('Confirmatory sampling blocked: '+', '.join(gate['blockers']))
    if not pool:raise ValueError('Empty candidate pool')
    validate_manifest(development)
    old_ids={(r['dataset'],r['image_id']) for r in development};old_hash={r['image_sha256'] for r in development}
    tasks=sorted({r['dataset'] for r in pool})
    if tasks!=['gqa','refcocog']:raise ValueError('Two fixed task pools required')
    rng=random.Random(seed);result=[]
    for task in tasks:
        eligible=[r for r in pool if r['dataset']==task and (r['dataset'],r['image_id']) not in old_ids and r['image_sha256'] not in old_hash]
        rng.shuffle(eligible)
        if len(eligible)<per_task:raise ValueError('Insufficient image-disjoint pool')
        result += [{**r,'partition':'confirmatory','seed':seed} for r in eligible[:per_task]]
    validate_manifest(development+result)
    return {'items':result,'manifest_sha256':digest(result),'seed':seed,'readiness':asdict(readiness),'status':'SAMPLED_AFTER_VALIDATED_LOCK'}


def validate_result_row(row):
    required={'task','model','image_id','candidate_correct','baseline_correct','protocol_sha256','manifest_sha256'}
    if not required<=row.keys():raise ValueError('Missing confirmatory result fields')
    if row['task'] not in ['gqa','refcocog'] or row['candidate_correct'] not in [0,.5,1] or row['baseline_correct'] not in [0,.5,1]:raise ValueError('Invalid paired result')
    if any(len(row[k])!=64 or any(c not in '0123456789abcdef' for c in row[k]) for k in ['protocol_sha256','manifest_sha256']):raise ValueError('Immutable hashes required')
    return True
