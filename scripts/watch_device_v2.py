"""Independent NVML observer during real stage execution; no model forwards."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import pynvml
from src.storage import atomic_json
from src.session import check_window


def run(seconds=900):
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0);rows=[];start=time.monotonic();last=None
    path=Path('outputs/v2/device_observer.csv');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        fields=['elapsed_seconds','model_label','gpu_utilization_percent','memory_used_bytes','power_watts','temperature_celsius'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        while time.monotonic()-start<seconds:
            check_window();label='unclassified'
            try:
                ledger=json.loads(Path('reports/evidence/v2/execution_ledger.json').read_text())
                event=next(e for e in reversed(ledger['events']) if e['task']=='D E2 computational scoring')
                label=event.get('model','unclassified')+'/'+event.get('status','unknown')
            except (FileNotFoundError,ValueError,StopIteration):pass
            row={'elapsed_seconds':time.monotonic()-start,'model_label':label,'gpu_utilization_percent':pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                 'memory_used_bytes':pynvml.nvmlDeviceGetMemoryInfo(handle).used,'power_watts':pynvml.nvmlDeviceGetPowerUsage(handle)/1000,
                 'temperature_celsius':pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU)}
            rows.append(row);writer.writerow(row)
            if len(rows)%20==0:
                stream.flush();summary={}
                for key in sorted({r['model_label'] for r in rows}):
                    subset=[r for r in rows if r['model_label']==key]
                    summary[key]={'samples':len(subset),'mean_gpu_utilization_percent':float(np.mean([r['gpu_utilization_percent'] for r in subset])),
                        'p95_gpu_utilization_percent':float(np.quantile([r['gpu_utilization_percent'] for r in subset],.95)),
                        'mean_power_watts':float(np.mean([r['power_watts'] for r in subset])),'max_memory_used_bytes':max(r['memory_used_bytes'] for r in subset),
                        'max_temperature_celsius':max(r['temperature_celsius'] for r in subset)}
                atomic_json('reports/evidence/v2/device_observer_summary.json',{'status':'RUNNING','sample_interval_seconds':.5,'scope':'whole device incl desktop; label from latest E2 ledger event, not per-kernel or process-isolated attribution','elapsed_seconds':time.monotonic()-start,'by_label':summary})
            time.sleep(.5)
    record=json.loads(Path('reports/evidence/v2/device_observer_summary.json').read_text());record['status']='COMPLETE';atomic_json('reports/evidence/v2/device_observer_summary.json',record);pynvml.nvmlShutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=900);run(p.parse_args().seconds)
