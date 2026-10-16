"""Bounded subprocess recovery preserving atomic per-input checkpoints."""
import subprocess,time
from pathlib import Path


def classify_failure(text):
    lower=text.lower()
    if any(s in lower for s in ['hard forward/gpu-time ceiling','autonomous work window ended','manifest changed','checkpoint corrupted','hash mismatch','vram headroom policy']):return 'PERMANENT_PROTOCOL_OR_INTEGRITY'
    if any(s in lower for s in ['cuda out of memory','cuda error: unknown error','cuda error: initialization error']):return 'RECOVERABLE_DEVICE_PROCESS'
    if any(s in lower for s in ['winerror 5','permissionerror','connectionreseterror','temporarily unavailable','connection aborted','read timed out','timed out','temporary failure in name resolution']):return 'RECOVERABLE_TRANSIENT'
    return 'PERMANENT_OR_UNDIAGNOSED'


def supervise(command,log_directory,max_attempts=3,runner=subprocess.run,sleep=time.sleep,on_attempt=None):
    directory=Path(log_directory);directory.mkdir(parents=True,exist_ok=True);history=[]
    for attempt in range(max_attempts):
        log=directory/f'attempt_{attempt+1}.log'
        with log.open('w',encoding='utf-8') as stream:completed=runner(command,stdout=stream,stderr=subprocess.STDOUT)
        tail=log.read_text(encoding='utf-8',errors='replace')[-12000:]
        category='SUCCESS' if completed.returncode==0 else classify_failure(tail)
        row={'attempt':attempt+1,'returncode':completed.returncode,'category':category,'log':str(log).replace(chr(92),'/')}
        history.append(row)
        if on_attempt:on_attempt(row)
        if category=='SUCCESS':return history
        allowed=category.startswith('RECOVERABLE') and attempt+1<max_attempts
        if category=='RECOVERABLE_DEVICE_PROCESS' and attempt>=1:allowed=False
        if not allowed:raise RuntimeError(f'Stage failed ({category}); preserved log {log}')
        sleep(min(2**attempt,4))
    raise AssertionError('Unreachable')
