import json
import time
from pathlib import Path
import threading

import numpy as np
import psutil
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText

from ..storage import atomic_json, cache_key, checkpoint, digest, read_checkpoint
from ..scoring import choice_probabilities


class Budget:
    """Single process ledger, persisted BEFORE each real forward, including failures."""
    def __init__(self, path=Path("outputs/budget.json"), max_calls=12000, max_hours=24):
        from ..budget import BudgetState
        from datetime import datetime
        self.state=BudgetState(path,max_calls,max_hours,start_epoch=datetime.fromisoformat('2026-10-16T07:18:11+00:00').timestamp())

    @property
    def value(self): return self.state.value

    def forward(self, model, **inputs):
        self.state.reserve()
        start = time.perf_counter()
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();failed=False;device_seconds=0
        try:
            result = model(**inputs, use_cache=False, return_dict=True)
            end.record()
            torch.cuda.synchronize()
            device_seconds=begin.elapsed_time(end)/1000
            return result
        except Exception:
            failed=True
            raise
        finally:
            self.state.finish(time.perf_counter()-start,failed,device_seconds)


class Telemetry:
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.stop = threading.Event()
        self.peak_nvml_used = 0
        self.peak_process_rss = 0
        self.samples = 0

    def _sample(self):
        while not self.stop.is_set():
            self.peak_nvml_used = max(self.peak_nvml_used, self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            self.peak_process_rss = max(self.peak_process_rss, psutil.Process().memory_info().rss)
            self.samples += 1
            self.stop.wait(0.1)

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.result = {"wall_seconds": time.perf_counter()-self.start,
                       "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "nvml_peak_device_used_bytes": self.peak_nvml_used,
                       "peak_process_rss_bytes": self.peak_process_rss, "telemetry_samples": self.samples,
                       "nvml_scope": "whole device including Windows desktop", "sample_interval_seconds": 0.1}
        self.nvml.nvmlShutdown()


class ChoiceModel:
    def __init__(self, model_id, revision, family, long_edge=512):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Immutable SHA required")
        self.model_id, self.revision, self.family = model_id, revision, family
        from filelock import FileLock
        Path("outputs").mkdir(exist_ok=True)
        self.process_lock=FileLock("outputs/primary_model.lock",timeout=0)
        self.process_lock.acquire()
        self.long_edge = long_edge
        self.processor_config = {"long_edge": long_edge, "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1}
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        if family == "qwen":
            self.processor.image_processor.max_pixels = long_edge*long_edge
            cls = Qwen2_5_VLForConditionalGeneration
        elif family == "smol":
            cls = AutoModelForImageTextToText
        else:
            raise ValueError("Unknown primary model family")
        self.model = cls.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16,
                                        attn_implementation="sdpa", trust_remote_code=False,
                                        device_map={"": "cuda:0"}, low_cpu_mem_usage=True).eval()
        self.model.requires_grad_(False)
        self.processor_config["official_image_processor"] = self.processor.image_processor.to_dict()
        self.processor_config["processor_class"] = type(self.processor).__name__
        self.budget = Budget()

    def set_long_edge(self, long_edge):
        if long_edge not in (384,512,768,1024): raise ValueError("Unregistered exploratory resolution")
        self.long_edge=long_edge
        self.processor_config["long_edge"]=long_edge
        if self.family=="qwen": self.processor.image_processor.max_pixels=long_edge*long_edge
        self.processor_config["official_image_processor"]=self.processor.image_processor.to_dict()

    def forward_inputs(self,inputs):
        return self.budget.forward(self.model,**inputs)

    def encode(self, image, question, options):
        if len(options) != 4 or len(set(options)) != 4:
            raise ValueError("Four unique options required")
        if image is not None:
            image = image.convert("RGB").copy()
            image.thumbnail((self.long_edge, self.long_edge))
        prompt = question + "\n" + "\n".join(f"{a}. {b}" for a,b in zip("ABCD", options)) + "\nAnswer with only the letter A, B, C, or D."
        content=([{"type":"image"}] if image is not None else []) + [{"type":"text", "text":prompt}]
        messages = [{"role":"user", "content":content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[image] if image is not None else None, return_tensors="pt")
        self.last_preprocess={"native_input_size":list(image.size) if image is not None else None,
                              "tensor_shapes":{k:list(v.shape) for k,v in inputs.items()},
                              "image_token_count":int(sum((inputs['input_ids']==self.processor.tokenizer.convert_tokens_to_ids(token)).sum().item()
                                  for token in (["<|image_pad|>"] if self.family=='qwen' else ["<image>"]))) if image is not None else 0}
        inputs = {k: v.to(device="cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
        # Check each literal label appended to the exact official rendered template.
        tokenizer = self.processor.tokenizer
        prefix = tokenizer.encode(text, add_special_tokens=False)
        labels = []
        for label in "ABCD":
            combined = tokenizer.encode(text+label, add_special_tokens=False)
            if combined[:len(prefix)] != prefix or len(combined) != len(prefix)+1:
                raise RuntimeError("Multi-token/retokenized label: exact sequence adapter needed; cannot silently score first token")
            labels.append(combined[-1])
        if len(set(labels)) != 4:
            raise RuntimeError("Label token collision")
        return text, inputs, labels

    @torch.inference_mode()
    def score(self, image, question, options, verify=False, use_cache=True):
        prompt, inputs, label_ids = self.encode(image, question, options)
        image_hash = digest(image.convert("RGB").tobytes()+str(image.size).encode()) if image is not None else digest({"mode":"text_only"})
        key = cache_key(model_id=self.model_id, revision=self.revision, processor_revision=self.revision,
                        image_sha256=image_hash, prompt=prompt, options=options,
                        processor_config=self.processor_config, code_sha256=digest(Path(__file__).read_bytes()))
        path = Path("outputs/cache") / f"{key}.json"
        if path.exists() and use_cache and not verify:
            row = dict(read_checkpoint(path,key))
            row["cache_hit"] = True
            return row
        start = time.perf_counter()
        logits = self.forward_inputs(inputs).logits[0,-1,:].float()
        scores = logits[label_ids].cpu().numpy()
        probabilities = choice_probabilities(scores)
        row = {"run_key":key, "status":"OK", "model_id":self.model_id, "model_revision":self.revision,
               "processor_revision":self.revision, "image_sha256":image_hash, "prompt_sha256":digest(prompt),
               "label_ids":label_ids, "choice_logits":scores.tolist(), "probabilities":probabilities.tolist(),
               "prediction":"ABCD"[int(probabilities.argmax())], "input_tokens":inputs["input_ids"].numel(),
               "wall_seconds":time.perf_counter()-start, "cache_hit":False, "retry_count":0,
               "processor_config":json.loads(json.dumps(self.processor_config)), "edit_params":None, "exception":None}
        row["preprocessing"]=dict(self.last_preprocess)
        if verify:
            # Independent full-vocabulary log-softmax then restrict/normalize choices.
            full = torch.log_softmax(logits.double(),dim=-1)[label_ids].cpu().numpy()
            reference = np.exp(full-full.max()); reference /= reference.sum()
            row["manual_probability_max_abs_error"] = float(np.max(np.abs(reference-probabilities)))
            repeat = self.forward_inputs(inputs).logits[0,-1,label_ids].float().cpu().numpy()
            row["repeat_logits_max_abs_error"] = float(np.max(np.abs(repeat-scores)))
        checkpoint(path, key, row)
        return row
