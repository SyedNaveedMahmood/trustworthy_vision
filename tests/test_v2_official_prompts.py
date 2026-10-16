"""CPU-only checks against installed official processor/tokenizer snapshots."""
import json
from pathlib import Path
import pytest
from PIL import Image


@pytest.mark.parametrize('key',['qwen','smol'])
def test_official_multimodal_template_and_exact_choice_continuations(key):
    transformers=pytest.importorskip('transformers')
    pin=json.loads(Path('reports/evidence/source_inventory.json').read_text())['models'][key]
    try:p=transformers.AutoProcessor.from_pretrained(pin['model_id'],revision=pin['revision'],local_files_only=True)
    except OSError:pytest.skip('Pinned official processor unavailable locally')
    question='Which object is red?\nA. ball\nB. cup\nC. chair\nD. bag\nAnswer with only the letter A, B, C, or D.'
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}]
    rendered=p.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    assert question in rendered
    inputs=p(text=[rendered],images=[Image.new('RGB',(64,64),'red')],return_tensors='pt')
    assert inputs['pixel_values'].numel()>0 and inputs['input_ids'].ndim==2
    prefix=p.tokenizer.encode(rendered,add_special_tokens=False);labels=[]
    for label in 'ABCD':
        seq=p.tokenizer.encode(rendered+(' ' if key=='smol' else '')+label,add_special_tokens=False)
        assert seq[:-1]==prefix;labels.append(seq[-1])
    assert len(set(labels))==4


def test_archived_smol_vocabulary_audit_detects_separator_bug():
    path=Path('reports/evidence/v2/choice_mass_smol.json')
    if not path.exists():pytest.skip('Actual vocabulary audit absent')
    record=json.loads(path.read_text());assert len(record['rows'])==80
    assert all(r['four_label_vocabulary_mass']<.001 for r in record['rows'])
    assert all(r['unrestricted_argmax_token'].startswith('\u0120') and r['unrestricted_argmax_token'][1:] in 'ABCD' for r in record['rows'])
