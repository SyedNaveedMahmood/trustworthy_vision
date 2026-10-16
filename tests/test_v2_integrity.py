import copy
from pathlib import Path
import pytest
from PIL import Image
from src.edits.pilot import request
from src.storage import digest
from src.evaluation.integrity import validate_scoring_inputs


def fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);image=Image.new('RGB',(100,80),'red');image.save('canonical.png')
    item={'dataset':'fixture','image_id':'1','canonical_path':'canonical.png','canonical_sha256':digest(Path('canonical.png').read_bytes())}
    edits=[]
    # Fixture duplicates cheap-operation output bytes under operator identities to
    # test manifest validation only. It is not a LaMa semantic or execution test.
    for ri in range(4):
        for op in ['occlusion','blur','lama']:
            e=request(image,(20,20,50,50),'occlusion');edits.append({**e,'task':'fixture','image_id':'1','region_index':ri,'operator':op})
    return [item],edits


def test_integrity_verifies_complete_factorial_and_bytes(tmp_path,monkeypatch):
    items,edits=fixture(tmp_path,monkeypatch)
    assert validate_scoring_inputs(items,edits)['expected_score_rows']==25
    with pytest.raises(ValueError,match='Incomplete'):validate_scoring_inputs(items,edits[:-1])
    with pytest.raises(ValueError,match='duplicate'):validate_scoring_inputs(items,edits+[edits[0]])
    Path(edits[0]['removal_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):validate_scoring_inputs(items,edits)
