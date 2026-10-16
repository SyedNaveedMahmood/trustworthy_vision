"""Actual campaign budgets and measured model intervals; no theoretical GPU timings."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
from src.storage import atomic_json


def run():
    budget=json.loads(Path('outputs/budget.json').read_text());models={}
    for model in ['qwen','smol','internvl']:
        runs=[json.loads(Path(f'reports/evidence/v2/E2_pairs_{cohort}_{model}.json').read_text()) for cohort in ['corrected','extension']]
        t=[r['telemetry'] for r in runs];n=sum(len(r['rows']) for r in runs);wall=sum(x['wall_seconds'] for x in t)
        models[model]={'paired_score_rows':n,'instrumented_model_interval_seconds':wall,'score_rows_per_instrumented_second':n/wall,
            'nvml_peak_device_used_bytes':max(x['nvml_peak_device_used_bytes'] for x in t),'torch_peak_allocated_bytes':max(x['torch_peak_allocated_bytes'] for x in t),
            'torch_peak_reserved_bytes':max(x['torch_peak_reserved_bytes'] for x in t),'peak_host_rss_bytes':max(x['peak_process_rss_bytes'] for x in t),
            'scope':'E2 initial+extension interval includes model load/preprocess/scoring/checkpoint I/O; excludes prior input-integrity scan. NVML whole device incl desktop.'}
    before_timing=max(json.loads(Path(f'reports/evidence/v2/hard_gqa_{m}.json').read_text())['cumulative_budget']['forward_calls'] for m in ['qwen','smol','internvl'])
    lama=[]
    for path in sorted(Path('reports/evidence/v2').glob('lama_batch*.json')):
        record=json.loads(path.read_text());lama.append({'file':str(path).replace(chr(92),'/'),**record})
    result={'status':'MEASURED_FINAL_GPU_EXECUTION','timestamp_utc':datetime.now(timezone.utc).isoformat(),'hardware':'NVIDIA RTX4080SUPER16GB driver610.88',
        'budget':budget,'prior_session_calls':776,'new_session_calls':budget['forward_calls']-776,'remaining_forward_allowance':12000-budget['forward_calls'],
        'cuda_stream_timing_starts_after_call':before_timing,'cuda_stream_timed_call_count':budget['forward_calls']-before_timing,
        'timing_warning':'CUDA events measure elapsed stream time around synchronized forwards, not summed active kernel time. Earlier calls were not CUDA-event profiled. Conservative GPU-time upper bound charges the entire session wall plus prior179.802s.',
        'paired_models':models,'official_lama_batches':lama,'free_disk_bytes_now':shutil.disk_usage('.').free,
        'human_review_completed':0,'confirmatory_forwards':0}
    atomic_json('reports/evidence/v2/resource_summary.json',result);atomic_json('reports/evidence/v2/forward_budget_final.json',budget)


if __name__=='__main__':run()
