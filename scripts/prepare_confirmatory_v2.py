"""Write an unlocked readiness snapshot and cost projection; no dataset sampling."""
import json
from pathlib import Path
import yaml
from src.evaluation.confirmatory import Readiness,blockers
from src.storage import atomic_json,digest
from src.session import update


def run():
    budget=json.loads(Path('outputs/budget.json').read_text());pins=yaml.safe_load(Path('configs/runtime_v2.yaml').read_text())
    # Provisional development competence is insufficient for locking an amended task.
    readiness=Readiness()
    gates=blockers(readiness,budget['forward_calls'])
    files=[p for pattern in ['src/**/*.py','configs/exploratory_v2.yaml','configs/runtime_v2.yaml','docs/08_ESTIMATOR_AMENDMENT.md','docs/09_DEVELOPMENT_EXTENSION.md'] for p in Path('.').glob(pattern)]
    snapshot={'status':'NOT_LOCKED_DO_NOT_RUN_E5','model_pins':pins['models'],'parameter_coefficient':.5,'primary_comparator':None,
              'planned_original_samples_per_task':120,'budget_projection':gates,
              'source_sha256':{str(p).replace(chr(92),'/'):digest(p.read_bytes()) for p in files},
              'confirmatory_manifest_generated':False,'confirmatory_model_forwards':0,
              'required_work':['independent two-rater semantic audit with operator/control thresholds','resolve narrowed task scope and qualification','select one comparator using development only','resolve 25-image-factorial cumulative budget conflict','freeze and validate hashes/splits']}
    atomic_json('reports/evidence/v2/confirmatory_readiness.json',snapshot)
    update('H E5 readiness preparation',status='BLOCKED',**gates)


if __name__=='__main__':run()
