"""Record exploratory execution pins without modifying original runtime pins."""
import json,platform,subprocess,shutil
from pathlib import Path
import yaml
from src.storage import digest,atomic_json


def run():
    config=yaml.safe_load(Path('configs/exploratory_v2.yaml').read_text())
    pins=yaml.safe_load(Path('configs/runtime_pins.yaml').read_text())
    record={'protocol':config['protocol'],'status':'NOT_CONFIRMATORY_LOCKED','protocol_config_sha256':digest(config),
            'models':{**pins['models'],'internvl':{'model_id':'OpenGVLab/InternVL2_5-2B','revision':'573169ee54df216786bb9a189e9a32a060a008cf','processor_revision':'573169ee54df216786bb9a189e9a32a060a008cf','dynamic_tiles_max':4,'thumbnail':True,'tile_edge':448,'attention':'eager'}},
            'execution':{'dtype':'bfloat16','batch_size':1,'image_long_edge':768,'sequential_models':True,'editor_device':'cpu','torch':'2.7.1+cu128','transformers':'4.51.3','new_dependencies':{'timm':'1.0.15','einops':'0.8.1','sentencepiece':'0.2.0','hypothesis':'6.131.9'}},
            'editor':pins['editor']}
    record['execution']['primary_models_after_amendment10']=['qwen','smol']
    record['execution']['canonical_label_separators']={'qwen':'','smol':' ','internvl':''}
    record['execution']['memory_headroom_gib']=1
    record['execution']['model_admission_free_gib']={'qwen':10,'smol':7,'internvl':8}
    Path('configs/runtime_v2.yaml').write_text(yaml.safe_dump(record,sort_keys=False),encoding='utf-8')
    git=str(Path('../.tools/git/cmd/git.exe').resolve())
    environment={'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                 'initial_commit':'ef7c3e84211cfc3e424a5b9ec27adac109dfe31d','initial_branch':'pilot/controlled-evidence-v1','initial_worktree_clean':True,
                 'safe_update':'fetch origin; pull --ff-only; no remote changes at session start','current_commit':subprocess.check_output([git,'rev-parse','HEAD'],text=True).strip(),
                 'free_disk_bytes_now':shutil.disk_usage('.').free,'platform':platform.platform(),
                 'gpu_now':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],text=True).strip(),
                 'initial_free_disk_approx_gb':866,'initial_gpu':'NVIDIA RTX 4080 SUPER 16376 MiB driver 610.88'}
    atomic_json('reports/evidence/v2/session_environment.json',environment)
    inventory={str(p).replace(chr(92),'/'):digest(p.read_bytes()) for pattern in ['docs/*','src/**/*.py','tests/*.py','reports/evidence/*.json'] for p in Path('.').glob(pattern) if p.is_file()}
    for filename in ['CODING_AGENT_MASTER_PROMPT.md','README.md','configs/pilot_v1.yaml','configs/runtime_pins.yaml','REPRODUCIBILITY.md','reports/FINAL_VERDICT.md','reports/STATUS.md','reports/DEVIATIONS.md']:
        inventory[filename]=digest(Path(filename).read_bytes())
    atomic_json('reports/evidence/v2/inspection_inventory.json',{'scope':'Reviewed baseline protocol/source/evidence; hashes identify files present during this session. Current files may include session edits.','sha256':inventory})


if __name__=='__main__':run()
