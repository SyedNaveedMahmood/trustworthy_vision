import pytest
from src.evaluation.selectivity import risk_coverage,paired_aurc_bootstrap


def test_risk_coverage_orders_and_excess_risk():
    assert risk_coverage([1,0],[1,0])['aurc']==.25
    assert risk_coverage([1,0],[0,1])['aurc']==.75
    assert risk_coverage([1,0],[1,0])['excess_aurc']==0
    with pytest.raises(ValueError):risk_coverage([1,.5],[1,0])


def test_paired_selectivity_bootstrap_preserves_identical_methods():
    rows=[{'task':t,'model':m,'image_id':str(i),'correct':i%2,'candidate_score':i,'baseline_score':i} for t in ['gqa','refcocog'] for m in ['a','b'] for i in range(4)]
    result=paired_aurc_bootstrap(rows,replicates=100)
    assert result['candidate_minus_confidence_aurc']==0 and result['ci95']==[0,0]
    with pytest.raises(ValueError,match='Missing'):paired_aurc_bootstrap(rows[:-1])
