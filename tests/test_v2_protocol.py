import copy,json
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given,settings,strategies as st
from PIL import Image
from src.scoring.v2 import robust,mad,estimators
from src.edits.pilot import independent_control,overlap,area,request,grid_candidates
from src.presentation import present
from src.budget import BudgetState
from src.storage import atomic_json,read_checkpoint,checkpoint

vectors=st.lists(st.floats(min_value=-100,max_value=100,allow_nan=False,allow_infinity=False),min_size=3,max_size=3)


@given(vectors,st.floats(min_value=0,max_value=2,allow_nan=False,allow_infinity=False))
def test_mutual_control_penalty_cancellation_property(values,coefficient):
    a=np.array(values)
    assert robust(a,coefficient)-robust(-a,coefficient)==pytest.approx(2*np.median(a),abs=1e-9)


@given(vectors,vectors,vectors)
def test_shared_control_linear_pair_cancels(a,b,c):
    a,b,c=map(np.asarray,(a,b,c))
    assert np.mean(a-c)-np.mean(b-c)==pytest.approx(np.mean(a-b),abs=1e-9)


def test_corrected_nonmutual_dispersion_can_change_rank():
    a=[0,3,6];b=[1,2,2]
    assert robust(a,0)>robust(b,0)
    assert robust(a,1)<robust(b,1)


def test_shared_vector_control_is_not_scalar_median_translation():
    rng=np.random.default_rng(41)
    changed=False
    for _ in range(100):
        a,b,c=rng.normal(size=(3,3))
        if np.sign(np.median(a)-np.median(b))!=np.sign(np.median(a-c)-np.median(b-c)): changed=True
    assert changed


def test_three_estimators_match_hand_calculation():
    sig=lambda x:1/(1+np.exp(-np.array(x)))
    d=np.array([1,2,8.]);control=np.array([.2,.3,.5])
    result=estimators(.5,sig(-d),[.5]*3,sig(-control),[.5]*3)
    assert result['sham_median']==pytest.approx(2)
    assert result['robust_sham']==pytest.approx(2-.5*1.4826)
    assert result['robust_independent']==pytest.approx(robust(d-control))


def test_control_determinism_equal_area_and_nonoverlap():
    box=(10,10,30,40)
    a=independent_control((100,80),box,123);b=independent_control((100,80),box,123)
    assert a==b and overlap(a,box)==0 and area(a)==area(box)
    assert independent_control((100,80),box,124)!=a


def test_control_rejects_impossible_geometry():
    with pytest.raises(ValueError): independent_control((100,80),(0,0,100,80),1)


def test_control_function_cannot_accept_gold_or_model_output():
    with pytest.raises(TypeError): independent_control((100,80),(10,10,30,40),1,gold='A')


@pytest.mark.parametrize('op',['occlusion','blur'])
def test_halo_sham_restores_core_and_outside_unchanged(tmp_path,monkeypatch,op):
    monkeypatch.chdir(tmp_path)
    rng=np.random.default_rng(12);image=Image.fromarray(rng.integers(0,256,(80,100,3),dtype=np.uint8))
    first=request(image,(20,20,50,50),op)
    assert first['outside_mask_max_change']==0 and first['sham_core_max_change']==0
    assert request(image,(20,20,50,50),op)==first
    assert request(image,(21,20,51,50),op)['key']!=first['key']
    if op=='blur': assert request(image,(20,20,50,50),op,sigma=10)['key']!=first['key']


def test_corrupted_edit_checkpoint_refused(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);im=Image.new('RGB',(100,80),'red')
    result=request(im,(20,20,50,50),'occlusion')
    Path(result['removal_path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError): request(im,(20,20,50,50),'occlusion')


def test_answer_rotation_distinct_from_visual_relabeling():
    im=Image.new('RGB',(160,120),'red');item={'dataset':'refcocog','question':'Select the box matching this description: red object',
       'options':['box A','box B','box C','box D'],'gold':'A','boxes':[(2,30,30,80),(40,30,70,80),(80,30,110,80),(120,30,150,80)]}
    a=present(im,item,'legacy');b=present(im,item,'outer_rotation');c=present(im,item,'visual_rotation')
    assert a[0].tobytes()==b[0].tobytes() and a[2]!=b[2] and b[3]=='B'
    assert a[0].tobytes()!=c[0].tobytes() and a[2]==c[2] and c[3]=='B'
    assert item['gold']=='A'  # input metadata immutable


def test_fixed_grid_needs_only_image_dimensions():
    boxes=grid_candidates((640,480));assert len(boxes)==4
    assert all(0<=b[0]<b[2]<=640 and 0<=b[1]<b[3]<=480 for b in boxes)


def test_budget_enforces_count_and_persists_restart(tmp_path):
    p=tmp_path/'budget.json';b=BudgetState(p,max_calls=1,clock=lambda:100,start_epoch=100,prior_gpu_seconds=0)
    b.reserve();b.finish(.3,failed=True)
    assert b.value['forward_calls']==1 and b.value['failed_calls']==1
    resumed=BudgetState(p,max_calls=1,clock=lambda:101,start_epoch=100,prior_gpu_seconds=0)
    with pytest.raises(RuntimeError): resumed.reserve()
    assert resumed.value['forward_calls']==1


def test_budget_enforces_conservative_gpu_hours(tmp_path):
    b=BudgetState(tmp_path/'b.json',max_hours=1,clock=lambda:4000,start_epoch=0,prior_gpu_seconds=0)
    with pytest.raises(RuntimeError): b.reserve()
    assert not b.path.exists()


def test_existing_v1_records_remain_unmodified():
    # Git revision is the archival source; this test checks the actual original gate numbers.
    for model,expected in [('qwen',28),('smol',15)]:
        row=json.loads(Path(f'reports/evidence/E1_{model}.json').read_text())
        assert row['metrics']['refcocog']['clean']['correct']==expected


def test_hard_color_choices_have_one_semantic_type_when_available():
    p=Path('outputs/v2/gqa_hard_manifest.json')
    if not p.exists(): pytest.skip('Local downloaded annotation-derived manifest unavailable')
    palette={'black','white','red','green','blue','yellow','brown','orange','pink','purple','gray'}
    rows=json.loads(p.read_text());assert len(rows)==20
    for row in rows: assert len(set(row['options']))==4 and set(row['options'])<=palette
