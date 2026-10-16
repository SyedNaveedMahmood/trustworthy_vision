"""Atomic active-session ledger; UTC timestamps, explicit finite work window."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import shutil
from filelock import FileLock
from .storage import atomic_json

PATH=Path("reports/evidence/v2/execution_ledger.json")
DEADLINE="2026-10-16T09:18:11+00:00"


def git_revision():
    executable=shutil.which('git')
    fallback=Path('../.tools/git/cmd/git.exe')
    if executable is None and fallback.exists():executable=str(fallback.resolve())
    if executable is None:return 'UNAVAILABLE'
    try:return subprocess.check_output([executable,'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:return 'UNAVAILABLE'


def update(task, status="RUNNING", **details):
    PATH.parent.mkdir(parents=True,exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    with FileLock('outputs/session_ledger.lock'):
        return _update(task,status,**details)


def _update(task,status,**details):
    old=json.loads(PATH.read_text()) if PATH.exists() else {"started_utc":"2026-10-16T07:18:11+00:00","deadline_utc":DEADLINE,"events":[]}
    now=datetime.now(timezone.utc).isoformat()
    entry={"timestamp_utc":now,"task":task,"status":status,**details}
    old["events"].append(entry); old["current"]=entry
    if Path("outputs/budget.json").exists(): old["budget"]=json.loads(Path("outputs/budget.json").read_text())
    old["git_commit"]=git_revision()
    atomic_json(PATH,old)
    print(task,status,flush=True)
    return old


def check_window():
    if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE):
        raise RuntimeError("Autonomous work window ended: checkpoint and hand off")
