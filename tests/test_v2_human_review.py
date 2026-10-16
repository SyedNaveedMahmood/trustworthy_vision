import pytest
from src.evaluation.human_audit import assess_reviews


def test_blank_form_and_single_rater_cannot_pass():
    key=[{'review_id':'FIXTURE','operator':'blur','region_index':0,'condition':'removal'}]
    assert not assess_reviews(key,[])['passed']
    assert assess_reviews(key,[{'review_id':'FIXTURE','reviewer_id':'SYNTHETIC_TEST_ONLY','removal_validity':'yes'}])['status']=='HUMAN_AUDIT_PENDING'


def test_duplicate_judgments_cannot_fake_independence():
    key=[{'review_id':'FIXTURE','operator':'blur','region_index':0,'condition':'removal'}]
    row={'review_id':'FIXTURE','reviewer_id':'SYNTHETIC_TEST_ONLY','removal_validity':'yes'}
    with pytest.raises(ValueError,match='Duplicate'):assess_reviews(key,[row,row])


def test_uncertain_judgments_do_not_vanish_from_denominator():
    key=[{'review_id':str(i),'operator':'blur','region_index':0,'condition':'removal'} for i in range(30)]
    ratings=[{'review_id':str(i),'reviewer_id':rid,'removal_validity':'uncertain' if i<4 else 'yes'} for rid in ['SYNTHETIC_A','SYNTHETIC_B'] for i in range(30)]
    result=assess_reviews(key,ratings)
    assert not result['passed'] and result['status']=='HUMAN_AUDIT_PENDING'
    assert next(iter(result['groups'].values()))['consensus_validity_rate']==26/30


def test_incomplete_operator_panel_cannot_pass_with_two_perfect_raters():
    key=[{'review_id':str(i),'operator':'blur','region_index':0,'condition':'removal'} for i in range(30)]
    ratings=[{'review_id':str(i),'reviewer_id':rid,'removal_validity':'yes'} for rid in ['SYNTHETIC_A','SYNTHETIC_B'] for i in range(30)]
    result=assess_reviews(key,ratings)
    assert not result['passed'] and len(result['missing_operator_condition_groups'])==11
