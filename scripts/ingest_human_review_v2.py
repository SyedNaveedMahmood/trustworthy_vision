"""Read actual reviewer copies; templates are never counted as completed reviews."""
import csv,json
from pathlib import Path
from src.evaluation.human_audit import assess_reviews
from src.storage import atomic_json


def run():
    root=Path('outputs/v2/human_review_combined');key=json.loads((root/'PRIVATE_blinding_key.json').read_text());ratings=[]
    files=sorted(root.glob('review_form_reviewer*.csv'))
    for path in files:
        with path.open(newline='',encoding='utf-8-sig') as stream:ratings+=list(csv.DictReader(stream))
    result=assess_reviews(key,ratings);result['reviewer_csv_files_received']=len(files)
    atomic_json('reports/evidence/v2/human_audit_ingestion.json',result)
    print(result['status'],result['submitted_judgment_rows'])


if __name__=='__main__':run()
