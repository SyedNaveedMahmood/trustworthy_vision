# Reproducing the executed partial pilot

## Current amended exploratory execution (v2)

See `reports/STATUS.md`, `reports/MODEL_QUALIFICATION_V2.md` and amendments08–10. The original instructions below are preserved as a v1 reproduction recipe, including its now-invalid Smol bare-token screen. Corrected Smol uses space-prefixed answer continuations; Qwen/InternVL use their canonical bare continuations. Do not regenerate old results in place with new scoring and call them unchanged.

Use `configs/requirements-vlm-v2.lock.txt` for the BF16 GPU environment and `configs/requirements-analysis-v2.lock.txt` for CPU analysis. Preserve the isolated original LaMa environment. CUDA torch is pinned to2.7.1+cu128; the publisher wheel/hash setup remains below. InternVL additional dependencies: timm1.0.15, einops0.8.1, sentencepiece0.2.0; official pinned remote code only. No quantization. All commands below run from the repository root; raw datasets, images, weights and checkpoints remain ignored.

```powershell
.venv-core\Scripts\python.exe -m scripts.prepare_qualification_v2
.venv\Scripts\python.exe -m scripts.qualification_v2 --model qwen
# Historical bare-token Smol diagnostics require archived33720e1 in a separate checkout.
.venv-core\Scripts\python.exe -m scripts.prepare_hard_gqa_v2
.venv\Scripts\python.exe -m scripts.qualification_corrected_smol
.venv-core\Scripts\python.exe -m scripts.prepare_extension_v2 --purpose qualification_corrected
.venv\Scripts\python.exe -m scripts.run_fresh_qualification_v2 --model qwen
.venv\Scripts\python.exe -m scripts.run_fresh_qualification_v2 --model smol
.venv\Scripts\python.exe -m scripts.run_fresh_qualification_v2 --model internvl
.venv-core\Scripts\python.exe -m scripts.prepare_intervention_v2 --cohort corrected --mode pairs
.venv\Scripts\python.exe -m scripts.supervise_stage_v2 --stage E2 --model qwen --cohort corrected
.venv\Scripts\python.exe -m scripts.supervise_stage_v2 --stage E2 --model smol --cohort corrected
.venv\Scripts\python.exe -m scripts.supervise_stage_v2 --stage E2 --model internvl --cohort corrected
.venv-core\Scripts\python.exe -m scripts.analyze_interventions_v2
.venv\Scripts\python.exe -m src.cli run --config configs/exploratory_v2.yaml --stage E4
.venv\Scripts\python.exe -m scripts.run_fill_factorial_v2 --model qwen
.venv\Scripts\python.exe -m scripts.run_fill_factorial_v2 --model internvl
.venv-core\Scripts\python.exe -m scripts.analyze_ablations_v2
.venv-core\Scripts\python.exe -m scripts.prepare_human_review_v2
.venv-core\Scripts\python.exe -m scripts.prepare_extension_v2
.venv-core\Scripts\python.exe -m scripts.prepare_intervention_v2 --cohort extension --mode pairs
.venv\Scripts\python.exe -m scripts.supervise_stage_v2 --stage E2 --model qwen --cohort extension
.venv\Scripts\python.exe -m scripts.supervise_stage_v2 --stage E2 --model smol --cohort extension
.venv\Scripts\python.exe -m scripts.supervise_stage_v2 --stage E2 --model internvl --cohort extension
.venv-core\Scripts\python.exe -m scripts.prepare_confirmatory_v2
.venv\Scripts\python.exe -m pytest -q
```

The historical qualification command must be run on archived code in a separate checkout to reproduce its bare-token result; the current wrapper is corrected. Current scripts preserve existing rows and use separate filenames for corrected scoring. E2/E3/E4 are independently runnable under the explicit exploratory config. E2's CLI defaults to the intended Qwen+corrected-Smol pair after amendment10; E4's actual input-variant comparison uses Qwen+InternVL and is labeled accordingly. E5/lock/confirmatory preparation refuse without prerequisites; `prepare_confirmatory_v2` writes only an unlocked readiness/cost snapshot, never a manifest.

Stage supervision logs every attempt, restarts only bounded transient/device failures in a fresh process and resumes atomic completed rows; it does not change resolution/precision to hide errors. Integrity failures and compute/headroom ceilings stop. Three models are never loaded together. Reserve2GiB free before a forward and require1GiB after; conservative loading thresholds are10/7/8GiB free for Qwen/Smol/InternVL. These admission guards were added after observed peaks and are not retroactive claims about earlier execution. Cumulative model-call ceiling remains12,000. Code/config/image/prompt hashes invalidate cache entries; completed stage checkpoints prevent accidental repeated forwards. An intentional verification repeat is counted.

Published score JSONs can be analyzed without model weights or datasets. Dataset audits, panel generation and fresh inference require the ignored local artifacts. Keep `outputs/v2/human_review/PRIVATE_blinding_key.json` hidden from independent reviewers. Review forms contain zero invented judgments and regeneration refuses to overwrite populated human fields. All source image files remain local.

## Archived v1 instructions

Current verdict: **INCONCLUSIVE**, halted after the failed two-model competence screen. E2–E5 are unfinished; a `run`/`lock` refusal is not an implementation of those experiments.

Preserved locally: `datasets/` (selected images and annotations), `outputs/manifests/` (full prompts/options/boxes and image hashes), `outputs/cache/` (content-addressed per-input checkpoints), `outputs/budget.json` (actual forward ledger), `outputs/sources/` (source snapshots), `outputs/lama_smoke/` (official predictions and full failure logs), `third_party_weights/` and the local Hugging Face model cache. Do not add these directories to Git. Git contains small score records, ID/hash summaries, figures, dependency locks and reports.

From repository root in PowerShell, with uv and Git available:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv venv .venv-core --python 3.11
uv pip install --python .venv-core\Scripts\python.exe -r configs/requirements-analysis.lock.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m src.cli environment
.venv-core\Scripts\python.exe -m scripts.prepare_development
.venv\Scripts\python.exe -m src.cli validate-images --split debug
.venv\Scripts\python.exe -m src.cli validate-images --split development
.venv\Scripts\python.exe -m src.cli synthetic-smoke --model qwen
.venv\Scripts\python.exe -m src.cli synthetic-smoke --model smol
.venv\Scripts\python.exe -m src.cli run --stage E1
.venv-core\Scripts\python.exe -m scripts.analyze_executed
.venv-core\Scripts\python.exe -m scripts.write_reports
```

The existing debug/development manifests are preserved; validation checks source image bytes. Repeated scores reuse content hashes; manual-repeat verification deliberately performs extra forwards. Code changes invalidate affected cache keys, and new calls increment the persistent budget. These commands reproduce E0/E1 only, not the complete experiment. The CLI accepts the hashed original pilot protocol; changes require an implemented/documented amendment. Effective image resolution is 512px within the protocol's 768px cap.

For fresh dataset access, first run `python -m src.cli audit-sources`; download the official-issue-linked archive with Windows TLS verification:

```powershell
curl.exe -fLsS 'https://web.archive.org/web/20220413012904id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip' -o outputs/refcocog-archive-attempt.bin
curl.exe -fLsS 'https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip' -o outputs/sceneGraphs.zip
.venv-core\Scripts\python.exe scripts/inspect_sources.py
.venv-core\Scripts\python.exe -m scripts.prepare_debug
.venv-core\Scripts\python.exe -m scripts.prepare_development
```

Check archive and per-image hashes against the committed summaries. Original RefCOCOg host DNS failure, Python certificate-chain failure and repaired native download are documented. `scripts/download_cuda_wheel.py` repairs the CUDA download with official-host byte ranges and verified publisher SHA-256; use its resulting local wheel if the ordinary installer stalls. `scripts/download_models.py` fetches the immutable primary snapshots. No private Hugging Face token is needed for the two original models. Fallback PaliGemma2 access needs the account owner's manual model access approval; it was not run.

LaMa reproduction uses `configs/requirements-lama.lock.txt`, Python 3.8.20, torch 1.8.0+cpu, an exact checkout of `advimman/lama` at `786f5936b27fb3dacd2b1ad799e4de968ea697e7`, and author-linked `smartywu/big-lama` archive revision `05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9`. Install the exact CPU torch/torchvision/torchtext wheels from the official PyTorch CPU index, then the remaining lock from PyPI. Extract weights into `third_party_weights/big-lama`; run `.venv-core\Scripts\python.exe -m scripts.lama_smoke`. It invokes the unmodified source in a separate CPU process. `opencv-python-headless` replaces the duplicate GUI/headless collision. No GPU fit or semantic validity claim is made for LaMa.

Published E1 score records can be reanalyzed without datasets/models by installing the analysis lock and running `scripts.analyze_executed` then `scripts.write_reports`; `budget_snapshot.json` supplies the saved ledger if local outputs are absent. Inferential results must remain labeled development diagnostics. `configs/locked_protocol.yaml` intentionally does not exist: no confirmatory lock was achieved.
