# E0: computational smoke completed; full gate not passed

2026-10-16, Asia/Dhaka. Source base `853f02959495f2056fd3ffea56ecf9b7e8bf3d27`; execution pins in `configs/runtime_pins.yaml`.

Both frozen models ran sequentially on CUDA BF16, batch 1, 512px maximum long edge, using official Transformers 4.51.3 processors/templates. Per model: 12 synthetic color-shape fixtures, 12 unique GQA images, 12 unique RefCOCOg images, and 4 occluded variants/task. Synthetic fixtures never enter benchmark manifests. Both scored 12/12 on synthetic color fixtures.

Every returned four-choice likelihood was finite. Token boundaries were checked by appending A/B/C/D to the exact rendered template; each label was one distinct token. The independent full-vocabulary log-softmax calculation matched conditional choice probabilities to floating-point precision. Repeated first-item logits were identical. See per-item `evidence/*_synthetic_smoke.json`; initial overlays remain in `evidence/*_overlay_v0_smoke.json`. The larger-letter repair did not fix SmolVLM2's 3/12 debug grounding result. This precludes a full E0 PASS. E1 was a flagged follow-up competence diagnosis, not a declaration that E0 passed.

Official LaMa source and author-linked weights were pinned and hashed. Its unmodified prediction script ran on an isolated CPU environment after protobuf and OpenCV repairs. Two successful executions returned identical image hashes (see `evidence/lama_replay.json`). Synthetic inpainting success is not semantic validity on natural images; no human gate passed.

GQA official balanced validation questions and scene graphs, RefCOCOg archived official UMD annotations, and selected official images were used. Debug and development byte hashes are disjoint. The official Visual Genome COCO-identity mapping found no known overlap among 104 sampled images; 25 GQA images have null COCO mappings, so identity coverage is incomplete. See `evidence/cross_dataset_identity_check.json`.

| Model | Peak allocated GiB | Peak reserved GiB | NVML whole-device peak GiB | Peak host RSS GiB | E1 wall seconds |
|---|---:|---:|---:|---:|---:|
| qwen | 7.573 | 7.656 | 8.780 | 4.212 | 36.648 |
| smol | 4.574 | 5.258 | 6.379 | 5.196 | 86.551 |

NVML samples every 0.1s and includes the Windows desktop. Peak sampling can miss shorter transients; torch allocator peaks are exact for the instrumented processes. Physical capacity was measured as 16,376 MiB; all runs retained more than 1 GiB headroom. No quantization, OOM or failed model forward occurred.

27 core tests pass, including protocol YAML parsing, safe numerical logits, masks, resize geometry, label pairing, split leakage, cache tampering, pinned revisions, shared-image bootstrap, retry bounds, semantic-review flags and refusal to lock. Full result-row schema validation covered all 640 E1 rows. End-to-end E2–E5 implementations/tests remain unfinished.
