from types import SimpleNamespace
from pathlib import Path
import json
import pytest
from src.execution import supervise,classify_failure
from src.storage import atomic_json
from src.resources import check_memory,GIB


def test_supervisor_restarts_transient_and_preserves_completed_checkpoint(tmp_path):
    checkpoint=tmp_path/'done.json';calls=[];sleeps=[]
    def runner(command,stdout,stderr):
        if not checkpoint.exists():
            calls.append('forward');atomic_json(checkpoint,{'completed':True});stdout.write('PermissionError: WinError 5');return SimpleNamespace(returncode=1)
        assert json.loads(checkpoint.read_text())['completed'];return SimpleNamespace(returncode=0)
    history=supervise(['fixture'],tmp_path/'logs',runner=runner,sleep=sleeps.append)
    assert len(calls)==1 and len(history)==2 and sleeps==[1]


def test_supervisor_does_not_retry_integrity_or_hard_budget(tmp_path):
    calls=[]
    def runner(command,stdout,stderr):calls.append(1);stdout.write('Hard forward/GPU-time ceiling reached');return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError,match='PERMANENT_PROTOCOL'):supervise(['fixture'],tmp_path,runner=runner,sleep=lambda _:None)
    assert len(calls)==1


def test_oom_recovery_is_bounded_to_one_retry(tmp_path):
    calls=[]
    def runner(command,stdout,stderr):calls.append(1);stdout.write('CUDA out of memory');return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError):supervise(['fixture'],tmp_path,runner=runner,sleep=lambda _:None)
    assert len(calls)==2


def test_vram_guard_rejects_insufficient_free_headroom():
    assert check_memory(3*GIB,16*GIB,2*GIB,'fixture')
    with pytest.raises(MemoryError,match='VRAM headroom'):check_memory(GIB,16*GIB,2*GIB,'fixture')
