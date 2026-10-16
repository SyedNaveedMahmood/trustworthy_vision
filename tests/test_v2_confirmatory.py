from dataclasses import fields
import pytest
from src.evaluation.confirmatory import Readiness,blockers,generate_confirmatory_manifest,validate_result_row


def ready():return Readiness(**{f.name:True for f in fields(Readiness)})


def test_missing_human_audit_and_budget_are_independent_blocks():
    gate=blockers(Readiness(),1000)
    assert not gate['ready'] and 'operator_human_audit_passed' in gate['blockers'] and 'cumulative_forward_budget_infeasible' in gate['blockers']
    assert not blockers(ready(),1000)['ready']
    assert blockers(ready(),1000,samples_per_task=100)['ready']


def test_no_manifest_sampling_before_gates_even_with_empty_pool():
    with pytest.raises(RuntimeError,match='Confirmatory sampling blocked'):generate_confirmatory_manifest([],[],Readiness(),0)


def test_confirmatory_schema_requires_complete_hashes():
    row={'task':'gqa','model':'fixture','image_id':'1','candidate_correct':1,'baseline_correct':.5,'protocol_sha256':'a'*64,'manifest_sha256':'b'*64}
    assert validate_result_row(row)
    row['manifest_sha256']='main'
    with pytest.raises(ValueError):validate_result_row(row)


def test_amended_cli_lock_and_e5_refuse(monkeypatch):
    from src.cli import main
    for args in [['lock'],['run','--stage','E5'],['prepare-data','--split','confirmatory']]:
        monkeypatch.setattr('sys.argv',['cli','--config','configs/exploratory_v2.yaml']+args)
        assert main()==2
