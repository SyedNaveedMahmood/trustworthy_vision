import copy
import json
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image
import pytest

from src.storage import atomic_json, digest, checkpoint, read_checkpoint, cache_key, retry_network
from src.scoring import choice_probabilities, bounded_logit, evidence, pair_correct
from src.evaluation import paired_cluster_bootstrap
from src.edits import xywh_to_xyxy, resize_box, box_mask, edit, overlay_boxes
from src.data import validate_manifest


def test_conditional_likelihood_against_manual_reference():
    logits = np.array([-1.0, 0.0, 2.0, -2.0])
    expected = np.exp(logits)/sum(np.exp(logits))
    np.testing.assert_allclose(choice_probabilities(logits), expected, atol=1e-14)
    np.testing.assert_allclose(choice_probabilities(logits+10000), expected, atol=1e-12)


@pytest.mark.parametrize("bad", [[0,1,2], [0,1,2,float('nan')], [0,1,2,float('inf')]])
def test_reject_bad_choice_scores(bad):
    with pytest.raises(ValueError): choice_probabilities(bad)


def test_logit_clamps_are_counted():
    result, clamps = bounded_logit([0,0.5,1])
    assert np.isfinite(result).all() and clamps == 2
    assert result[1] == 0
    np.testing.assert_allclose(result[0], -result[2], atol=1e-9)


def test_corrected_equation_and_negative_score_not_clipped():
    result = evidence(.7,[.2,.4,.6],[.6,.5,.7],[.4,.5,.6],[.5,.6,.7])
    rem = np.log(np.array([.2,.4,.6])/(1-np.array([.2,.4,.6])))
    sham = np.log(np.array([.6,.5,.7])/(1-np.array([.6,.5,.7])))
    ctrl = np.log(np.array([.5,.6,.7])/(1-np.array([.5,.6,.7]))) - np.log(np.array([.4,.5,.6])/(1-np.array([.4,.5,.6])))
    z = sham-rem-ctrl
    m = np.median(z); v = 1.4826*np.median(abs(z-m))
    np.testing.assert_allclose(result["z"], z, atol=1e-14)
    assert result["score"] == pytest.approx(m-.5*v)
    negative = evidence(.5,[.8,.9,.7],[.2,.3,.1],[.5]*3,[.5]*3)
    assert negative["score"] < 0


def test_mutual_controls_cancel_dispersion_in_pair_contrast():
    # Mathematical diagnostic of the unresolved 13-image protocol interpretation.
    z = np.array([-1.,.5,4.])
    scores = []
    for coefficient in (0,.25,.5,1):
        def score(a):
            return np.median(a)-coefficient*1.4826*np.median(abs(a-np.median(a)))
        scores.append(score(z)-score(-z))
    np.testing.assert_allclose(scores,[1]*4)


def test_pairing_half_credit_and_direction():
    np.testing.assert_array_equal(pair_correct([3,1,2],[2,2,2]),[1,0,.5])
    with pytest.raises(ValueError): pair_correct([1,float('nan')],[2,3])


def paired_rows():
    return [{"task":task,"model":model,"image_id":str(i),"candidate_correct":(i%2),"baseline_correct":0}
            for task in ("gqa","refcocog") for model in ("qwen","smol") for i in range(8)]


def test_bootstrap_preserves_shared_model_image_clusters():
    rows = paired_rows()
    a = paired_cluster_bootstrap(rows)
    # Reference manual bootstrap: the two identical model outcomes must not be resampled independently.
    rng = np.random.default_rng(20264)
    expected = np.zeros(2000)
    outcomes = np.arange(8)%2
    for _ in range(2): expected += outcomes[rng.integers(0,8,(2000,8))].mean(axis=1)/2
    np.testing.assert_allclose(a["ci95"],np.quantile(expected,[.025,.975]))
    assert a["effect"] == .5
    assert a == paired_cluster_bootstrap(rows)


@pytest.mark.parametrize("rows", [[],paired_rows()[:-1],paired_rows()+[paired_rows()[0]]])
def test_bootstrap_rejects_empty_missing_and_duplicate_pairs(rows):
    with pytest.raises(ValueError): paired_cluster_bootstrap(rows)


def test_box_clip_resize_and_half_open_pixel_geometry():
    assert xywh_to_xyxy([-2,-3,12,13],20,20) == (0,0,10,10)
    assert resize_box((0,2,10,12),(20,20),(40,10)) == (0,1,20,6)
    mask = np.asarray(box_mask((20,20),(0,0,10,10)))
    assert np.count_nonzero(mask)==100 and mask[9,9]==255 and mask[10,10]==0


@pytest.mark.parametrize("box", [[0,0,-1,3],[30,30,2,2],[float('nan'),0,2,3]])
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError): xywh_to_xyxy(box,20,20)


def test_edits_do_not_change_off_mask_pixels_and_sham_keeps_interior():
    rng = np.random.default_rng(123)
    image = Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8))
    original = np.asarray(image)
    mask = np.asarray(box_mask(image.size,(10,10,40,40))).astype(bool)
    for operator in ("occlusion","blur"):
        for sham in (False,True):
            output = np.asarray(edit(image,(10,10,40,40),operator,sham))
            np.testing.assert_array_equal(output[~mask],original[~mask])
    sham = np.asarray(edit(image,(10,10,40,40),"occlusion",True))
    np.testing.assert_array_equal(sham[15:35,15:35],original[15:35,15:35])
    # This checks pixels, not semantic preservation in natural images.
    with pytest.raises(RuntimeError): edit(image,(10,10,40,40),"lama")


def test_box_overlays_apply_equally_to_all_conditions():
    boxes = [(2,2,12,12),(16,2,26,12),(2,16,12,26),(16,16,26,26)]
    image = Image.new("RGB",(32,32),"red")
    assert overlay_boxes(image,boxes).tobytes() == overlay_boxes(image.copy(),boxes).tobytes()


def manifest_row():
    return {"dataset":"gqa","image_id":"1","image_sha256":"a"*64,"image_path":"gqa/1.jpg",
            "partition":"debug","options":["red","blue","green","yellow"],"gold":"A",
            "source_url":"https://example.invalid","source_license":"fixture","audit_status":"NOT_REVIEWED",
            "annotation_ids":["1"],"task_subtype":"attribute","seed":20261}


def test_manifest_rejects_id_leakage_and_cross_dataset_duplicate_hashes():
    a = manifest_row(); b = copy.deepcopy(a); b["partition"]="confirmatory"
    with pytest.raises(ValueError): validate_manifest([a,b])
    b["dataset"]="refcocog"; b["image_id"]="2"
    with pytest.raises(ValueError): validate_manifest([a,b])


def test_manifest_checks_exact_image_bytes(tmp_path):
    a = manifest_row(); a["image_path"]="image.jpg"; data=b"fixture"; a["image_sha256"]=digest(data)
    (tmp_path/"image.jpg").write_bytes(data)
    assert validate_manifest([a],tmp_path)["count"]==1
    (tmp_path/"image.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError): validate_manifest([a],tmp_path)


def test_no_human_validation_without_two_raters():
    row=manifest_row(); row["audit_status"]="HUMAN_VALIDATED"
    with pytest.raises(ValueError): validate_manifest([row])
    row["human_rater_ids"]=["same","same"]
    with pytest.raises(ValueError): validate_manifest([row])


def test_no_unknown_semantic_validity_flag():
    row=manifest_row(); row["audit_status"]="AUTO_PASSED_AS_HUMAN"
    with pytest.raises(ValueError): validate_manifest([row])


def test_checkpoint_tampering_rejected(tmp_path):
    path = tmp_path/"result.json"
    checkpoint(path,"abc",{"value":[1,2]})
    assert read_checkpoint(path,"abc")=={"value":[1,2]}
    with pytest.raises(ValueError): read_checkpoint(path,"wrong")
    row=json.loads(path.read_text()); row["payload"]["value"][0]=3; atomic_json(path,row)
    with pytest.raises(ValueError): read_checkpoint(path,"abc")


def test_cache_key_covers_all_experimental_factors():
    args = {"model_id":"fixture","revision":"a"*40,"processor_revision":"a"*40,"image_sha256":"b"*64,
            "prompt":"question","options":["a","b","c","d"],"processor_config":{"resolution":512},"code_sha256":"c"*64}
    key=cache_key(**args)
    for field in args:
        altered=copy.deepcopy(args)
        altered[field] = "d"*40 if field in ("revision","processor_revision") else {"changed":True}
        assert cache_key(**altered)!=key
    args["revision"]="main"
    with pytest.raises(ValueError): cache_key(**args)


def test_network_retry_is_bounded_and_deterministic_errors_not_retried():
    calls=[]; sleeps=[]
    def transient():
        calls.append(1)
        if len(calls)<3: raise URLError("temporary")
        return "ok"
    assert retry_network(transient,sleep=sleeps.append)==("ok",2)
    assert len(calls)==3 and sleeps==[1,2]
    calls.clear()
    def permanent():
        calls.append(1); raise HTTPError("url",404,"missing",{},None)
    with pytest.raises(HTTPError): retry_network(permanent,sleep=sleeps.append)
    assert len(calls)==1


def test_confirmatory_lock_refused_before_prerequisites():
    from src.cli import main
    import sys
    old=sys.argv
    try:
        sys.argv=["cli","lock"]
        assert main()==2
    finally: sys.argv=old


def test_protocol_yaml_is_parseable_and_preserves_scientific_values():
    from pathlib import Path
    import yaml
    config=yaml.safe_load(Path('configs/pilot_v1.yaml').read_text(encoding='utf-8'))
    assert [o['id'] for o in config['operators']['panel']]==['occlusion','blur','lama']
    assert config['sample_plan_per_task']['development']==40
    assert config['sample_plan_per_task']['confirmatory']==120
    assert config['run_budget']['total_forward_calls_max']==12000
    assert config['run_budget']['gpu_hours_max']==24
