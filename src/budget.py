"""Persistent hard forward budget with conservative GPU wall-time accounting."""
import json,time
from pathlib import Path
from filelock import FileLock
from .storage import atomic_json


class BudgetState:
    def __init__(self,path=Path('outputs/budget.json'),max_calls=12000,max_hours=24,clock=time.time,start_epoch=None,prior_gpu_seconds=179.80193130001135):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.max_calls=max_calls; self.max_seconds=max_hours*3600;self.clock=clock
        self.start_epoch=clock() if start_epoch is None else start_epoch; self.prior_gpu_seconds=prior_gpu_seconds
        self.lock=FileLock(str(self.path)+'.lock',timeout=5)

    @property
    def value(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {'forward_calls':0,'forward_wall_seconds':0,'failed_calls':0,'cuda_stream_seconds':0}

    def reserve(self):
        with self.lock:
            value=self.value
            upper=self.prior_gpu_seconds+max(0,self.clock()-self.start_epoch)
            if value['forward_calls']>=self.max_calls or upper>=self.max_seconds:
                raise RuntimeError('Hard forward/GPU-time ceiling reached')
            value['forward_calls']+=1
            value['gpu_time_conservative_upper_bound_seconds']=upper
            value['gpu_time_bound_scope']='prior measured model-process wall + entire current autonomous-session wall; conservative, not active kernel time'
            atomic_json(self.path,value)

    def finish(self,seconds,failed=False,cuda_seconds=0):
        with self.lock:
            value=self.value;value['forward_wall_seconds']+=seconds;value['failed_calls']+=int(failed)
            value['cuda_stream_seconds']=value.get('cuda_stream_seconds',0)+cuda_seconds
            atomic_json(self.path,value)
