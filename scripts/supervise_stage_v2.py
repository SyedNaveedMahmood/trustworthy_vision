"""Run resumable E2/E4 child processes with bounded recoverable-failure retries."""
import argparse,sys
from src.execution import supervise
from src.session import update,check_window


def run(stage,model,cohort='corrected',mode='pairs'):
    check_window()
    module='scripts.score_interventions_v2' if stage=='E2' else 'scripts.run_ablations_v2'
    command=[sys.executable,'-m',module,'--model',model]
    if stage=='E2':command+=['--cohort',cohort,'--mode',mode]
    return supervise(command,f'outputs/v2/supervision/{stage}_{cohort}_{mode}_{model}',on_attempt=lambda row:update('G stage supervision',stage=stage,model=model,**row))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['E2','E4'],required=True);p.add_argument('--model',choices=['qwen','internvl','smol'],required=True)
    p.add_argument('--cohort',choices=['corrected','extension'],default='corrected');p.add_argument('--mode',choices=['pairs','grid'],default='pairs');a=p.parse_args();run(a.stage,a.model,a.cohort,a.mode)
