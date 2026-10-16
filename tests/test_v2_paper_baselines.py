import pytest
from src.scoring.paper_baselines import ced,answer_tokens
from scripts.run_ablations_v2 import scaled_box


def test_ced_equations_normalize_semantic_answers():
    assert ced('A RED dog!','a red dog')['ced']==0
    assert ced('red','blue')['ced']==1
    assert ced('red car','red bus')['ced']==pytest.approx(2/3)
    assert ced('one two three four','one two three')['ced']==0
    assert ced('one two three four','one two three')['ced_soft']==.25
    assert ced('','')['ced']==0


def test_scaled_mask_coordinates_clipped_without_translation():
    assert scaled_box([20,20,60,60],(100,100),.5)==[30,30,50,50]
    assert scaled_box([0,0,20,20],(100,100),2)==[0,0,30,30]
