import copy,json
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given,strategies as st
from src.evaluation.v2 import auc,pair_credit,paired_summary,paired_difference
from scripts.analyze_interventions_v2 import extract
from src.storage import atomic_json,digest
from src.edits.pilot import request
from PIL import Image


def test_auc_known_ties_and_opposite_ranking():
    assert auc([2,3],[0,1])==1
    assert auc([0,1],[2,3])==0
    assert auc([1,1],[1,1])==.5
    assert pair_credit(1,1)==.5


@given(st.lists(st.floats(-20,20,allow_nan=False,allow_infinity=False),min_size=1,max_size=30))
def test_paired_bootstrap_identical_methods_have_exact_zero_interval(values):
    result=paired_difference(values,values,replicates=100)
    assert result['effect']==0 and result['ci95']==[0,0]


def test_bootstrap_preserves_pairing_and_determinism():
    a=paired_summary([2,3,4],[1,2,3],replicates=100)
    assert a==paired_summary([2,3,4],[1,2,3],replicates=100)
    assert a['ranking_ci95']==[1,1]
    with pytest.raises(ValueError):paired_summary([1,2],[1])


def fixture_record():
    rows=[{'task':'gqa','image_id':'1','variant':'clean','probabilities':[.6,.2,.1,.1],'prediction':'A','gold':'B','correct':False}]
    for ri in range(4):
        for op in ['occlusion','blur','lama']:
            for condition in ['removal','sham']:
                rows.append({**rows[0],'variant':f'r{ri}_{op}_{condition}','probabilities':[.1,.6,.2,.1],'prediction':'B'})
    return {'rows':rows}


def test_frozen_prediction_anchor_does_not_follow_edited_argmax():
    result=extract(fixture_record())
    assert next(r for r in result if r['anchor']=='predicted')['anchor_letter']=='A'
    assert next(r for r in result if r['anchor']=='gold')['anchor_letter']=='B'


def test_result_schema_missing_duplicate_and_nonfinite_refused():
    record=fixture_record();record['rows'].pop()
    with pytest.raises(ValueError,match='Incomplete'):extract(record)
    record=fixture_record();record['rows'].append(record['rows'][0])
    with pytest.raises(ValueError,match='Duplicate'):extract(record)
    record=fixture_record();record['rows'][0]['probabilities']=[np.nan,.2,.3,.4]
    with pytest.raises(ValueError,match='Invalid'):extract(record)


def test_lama_output_shape_checked_without_claiming_real_model_execution(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);im=Image.new('RGB',(100,80),'red')
    first=request(im,(20,20,50,50),'lama');out=Path('outputs/v2/lama/output');out.mkdir(parents=True)
    Image.new('RGB',(99,80)).save(out/f"{first['key']}_mask001.png")
    with pytest.raises(ValueError,match='unpadding'):request(im,(20,20,50,50),'lama')


def test_image_content_and_fill_invalidate_edit_key(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);a=Image.new('RGB',(100,80),'red');b=a.copy();b.putpixel((0,0),(0,0,0))
    first=request(a,(20,20,50,50),'occlusion')
    assert request(b,(20,20,50,50),'occlusion')['key']!=first['key']
    assert request(a,(20,20,50,50),'occlusion',fill='neutral')['key']!=first['key']


def test_source_split_identity_summary_when_present():
    path=Path('reports/evidence/v2/dataset_audit.json')
    if not path.exists():pytest.skip('Executed split audit absent')
    audit=json.loads(path.read_text())['split_integrity']
    assert audit['hard_duplicates']==[] and audit['perceptual_near_duplicates']==[]
