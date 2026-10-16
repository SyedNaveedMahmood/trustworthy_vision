# Model qualification V2

**Provisional choice: Qwen2.5-VL-3B + InternVL2.5-2B.** The unchanged engineering screen is 40% image accuracy and positive image-versus-text gain. This is a small-sample feasibility screen, not a statistical proof of visual reasoning. SmolVLM2 remains unqualified on independent grounding; it was not discarded solely because of its original one-example deficit.

## Grounding development diagnostics

All conditions share the original 40 development images. Each comparison changes resolution or presentation alone relative to its stated reference. Legacy 512→768 scores: Qwen 28→30; Smol 15→18. 1024 gives the same accuracies: native COCO sources are at most 640 pixels, and the wrapper does not upscale them. The official processor then performs its own resizing/patching. Consequently, requested cap is not actual image detail or token count.

| Model | Development condition | Correct / 40 | Prediction A/B/C/D | Mean image tokens | Mean score seconds |
|---|---|---:|---|---:|---:|
| qwen | legacy_1024_image | 30 | 12/8/8/12 | 336.9 | 0.1425 |
| qwen | legacy_512_image | 28 | 11/7/11/11 | 230.1 | 0.1135 |
| qwen | legacy_768_image | 30 | 12/8/8/12 | 336.9 | 0.1562 |
| qwen | numbered_768_image | 31 | 13/4/11/12 | 336.9 | 0.1503 |
| qwen | numbered_768_text_only | 5 | 30/10/0/0 | 0.0 | 0.0312 |
| qwen | outer_rotation_768_image | 26 | 15/13/6/6 | 336.9 | 0.1351 |
| qwen | visual_rotation_768_image | 30 | 14/8/8/10 | 336.9 | 0.1430 |
| smol | legacy_1024_image | 18 | 3/21/8/8 | 1093.5 | 0.2046 |
| smol | legacy_512_image | 15 | 3/20/9/8 | 1093.5 | 0.2111 |
| smol | legacy_768_image | 18 | 3/21/8/8 | 1093.5 | 0.2041 |
| smol | numbered_768_image | 16 | 3/21/6/10 | 1093.5 | 0.2033 |
| smol | numbered_768_text_only | 11 | 0/13/0/27 | 0.0 | 0.0266 |
| smol | outer_rotation_768_image | 13 | 1/10/14/15 | 1093.5 | 0.2029 |
| smol | visual_rotation_768_image | 15 | 3/25/6/6 | 1093.5 | 0.2041 |

Outer-letter rotation changes text answer positions and remaps the gold answer while keeping visual letters fixed. Visual rotation changes labels in the image and keeps the A–D text ordering. Numbered regions use inner 1–4 and outer A–D. These distinguish binding from position effects; none provides a clear, independently validated fix for Smol. B frequency alone cannot establish a token-scoring defect: exact label continuations are checked and manual probability and repeat checks passed. Correct/incorrect examples are preserved locally at `outputs/v2/diagnostic_panels/`.

## Independent qualification

20 RefCOCOg images are disjoint from original debug/development and were never used for selecting presentation. At legacy 768: Qwen **11/20 vs 7/20 text-only**; Smol **7/20 vs 5/20**; InternVL **12/20 vs 7/20**. Smol fails the fixed 8/20 minimum. These n=20 estimates are imprecise; refer to saved item-level outputs rather than extrapolating population competence.

The original GQA qualification reused an invalid mixed-type distractor recipe and is retained only as a diagnostic. The separate annotation-selected, image-disjoint basic-color set gives Qwen **16/20 image, 6/20 text, 4/20 shuffled**; InternVL **15/20, 7/20, 5/20**; Smol **15/20, 7/20, 8/20**. This qualifies a narrow color-identification development task; broad GQA competence remains unestablished. Its selection did not inspect model outputs, but it is development evidence, not confirmation.

## Official fallback and resources

[Official InternVL checkpoint](https://huggingface.co/OpenGVLab/InternVL2_5-2B), pinned revision `573169ee54df216786bb9a189e9a32a060a008cf`; model card declares MIT and embedded InternLM code carries Apache-2.0 headers. Source and weight hashes are recorded in `internvl_source_probe.json` and `internvl_download.json`. Official dynamic tiling is adapted with a fixed maximum four tiles plus thumbnail, 448 pixels; no quantization, BF16, eager attention because FlashAttention is absent. Qwen/Smol retain their official processors and SDPA. Models run sequentially under a process lock. The 4-tile cap is an explicit compute adaptation, not an official benchmark reproduction. No gated PaliGemma retry occurred.

Full-process peaks and times are in the machine-readable summary; image-token counts and original per-score wall times are in the table. Development Smol resume telemetry covers only the final resumed process and is not a reliable full-grid peak. NVML includes Windows desktop memory; Torch allocated/reserved peaks describe the process allocator. InternVL qualification measured 8.04 GiB device usage, leaving substantial headroom on the 16 GiB card.

**Gate assessment:** two distinct backbones meet the provisional corrected-task engineering screen. This does not clear semantic intervention validation, independent human audit, amended confirmatory lock, or full E0/E1 scientific gates. E5 remains blocked.
