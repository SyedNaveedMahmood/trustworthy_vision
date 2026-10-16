# 07 — Single RTX 4080 SUPER 16 GB deployment plan

All performance and fit statements here are a DESIGN ESTIMATE. Nothing has yet been executed on this user's GPU. The coding agent must measure hardware and adjust before a full run.

## Baseline model selection

- Primary: Qwen/Qwen2.5-VL-3B-Instruct; public model listing indicates ~7.52 GB of weights on disk, excluding activations and preprocessing. Try BF16, a single GPU, batch size 1.
- Independent second family: HuggingFaceTB/SmolVLM2-2.2B-Instruct; source model card shows two weight shards, collectively ~9 GB on disk; again disk weight size does not predict peak VRAM precisely. Try BF16 batch size 1, controlled image resolution.
- Do NOT load both concurrently. Free model weights and collect peak memory after each run. Avoid evaluation-time gradient tracking.
- Use the official chat templates and a verified compatible Transformers release. Models may need different processor APIs.
- Keep max image long edge / effective pixel cap configurable and recorded. Start with 512-768px depending on dataset needs; high resolution and large text prompts may dramatically increase image-token count.
- Prefer CUDA BF16 torch.inference_mode(). If 16GB OOM, first reduce effective image resolution, image patches, batch size and concurrent editor processes; then consider verified 4-bit quantization, with **precision as a named experimental factor** and all model scores recomputed under one fixed precision. Never silently mix BF16 and 4-bit results.

## Memory handling

Aim for at least ~1 GiB headroom; log torch.cuda.max_memory_reserved(), max_memory_allocated(), NVML peak reserved/used and host RAM. 16 GB is the physical VRAM ceiling, not an invitation to assume 16 GB is fully free. Editors such as LaMa must run sequentially or in an isolated subprocess; save edited images to disk, exit the edit process and then load a VLM.

The 13-image factorial multiplied by 240 confirmatory image-items and two models implies a lower-bound of 6,240 model-image forward-scoring events, BEFORE permutations, robustness variants and failed attempts. Teacher-forced scoring of four answers may use shared-prefix forward passes when mathematically equivalent. Count actual forward calls and tokens, not speculative time-to-finish.

## Staged commands to implement

    python -m src.cli audit-sources --config configs/pilot_v1.yaml
    python -m src.cli prepare-data --config configs/pilot_v1.yaml --split debug
    python -m src.cli validate-images --config configs/pilot_v1.yaml --split debug
    python -m src.cli score --config configs/pilot_v1.yaml --split debug --model qwen
    python -m src.cli score --config configs/pilot_v1.yaml --split debug --model smol
    python -m src.cli run --config configs/pilot_v1.yaml --stage E1
    python -m src.cli run --config configs/pilot_v1.yaml --stage E2
    python -m src.cli run --config configs/pilot_v1.yaml --stage E3
    python -m src.cli lock --config configs/pilot_v1.yaml
    python -m src.cli run --config configs/pilot_v1.yaml --stage E5
    python -m src.cli report --config configs/pilot_v1.yaml

These commands are DESIRED CLI CONTRACTS, not currently implemented. Agent may choose equivalent invocations but must document real commands. Do not report that the command exists until implemented and tested.

## Compute and failure budget

- Perform first true-device benchmark on 12 debug images/task/model; measure median and p95 scoring time, GPU peak and editor throughput.
- Project total wall time from measured calls and cache hit rate. Record the estimate and revise from checkpointed actuals; no invented minutes.
- Hard default: max 12,000 model-scoring forward operations over all pilot stages and no more than 24 cumulative GPU hours; this is a configurable experimental ceiling, not an expected runtime. If exceeded, STOP WITH PARTIAL REPORT and request a scoped redesign instead of silently expanding the study.
- Retry transient network/cache/editor errors up to 3 times with exponential backoff; do not retry deterministic CUDA OOM without modifying configuration and documenting changes.
- All output rows should have run ID, model and processor revision, image hash, prompt hash, editor params, seeds, prediction logits, status, exception and retry count.
- Use atomic write-then-rename for cached outputs. Results must be resumable from completed (model, image, operator, region, prompt, options) tuples. Deduplicate by run key; hashes must match before cache reuse.
- Do NOT cache credentials, absolute private paths or full dataset images in GitHub. Verify .gitignore excludes models/, datasets/, third_party_weights/, outputs/ and .cache/.

## Gatekeeper checklist

G0: reproducible external source inventory, download/weights licenses, hashes and dependency compatibility; no fake "official code".

G1: both frozen models load separately, return comparable finite 4-choice scores, masks align, all 24 debug items complete, 16GB memory respected.

G2: per-model/task dev accuracy above the predeclared gate and nontrivial image-dependent behavior. If not, downgrade or replace the model transparently.

G3: edit validity audit (especially sham-preservation) complete or explicitly PENDING; bad operators cannot be used to justify semantic claims.

G4: proposed score beats strongest dev comparator by preregistered practical margin without excessive extra computation; otherwise conclude negative pilot.

G5: confirmatory protocol locked and tested as read-only; do not reopen it for algorithmic tuning.

G6: confirmatory results, uncertainties, failure analysis and code/report reproducibility; verdict GO / NO-GO / INCONCLUSIVE.

The coding agent must write reports/STATUS.md at each gate and report numbers only when produced by actual files/logs. No background execution is implied by writing this document.
