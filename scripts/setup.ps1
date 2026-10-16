$ErrorActionPreference = 'Stop'
# Requires uv on PATH; creates only the repository environment.
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA torch install failed' }
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed' }
