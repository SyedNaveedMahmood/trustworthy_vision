# Reproducing the executed partial pilot

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
