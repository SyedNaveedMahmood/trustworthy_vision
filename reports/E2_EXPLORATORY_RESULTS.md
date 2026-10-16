# E2 exploratory results — completed computational development

**81 real images, three qualified frozen backbones,6,075 complete paired score rows; HUMAN_AUDIT_PENDING.** Initial15 GQA/16 RefCOCOg plus25 new images/task. Every image has clean+24 edits per model. All3 backbones use identical cached edited images; no confirmatory sample was accessed.

972 content/mask/operator requests produce 1944 removal/sham images. Mean-fill occlusion, Gaussian blur sigma24 and official LaMa share canonical RGB capped at768 and a fixed8px halo. Shams restore exact original core pixels; outside-mask compositing is exact. Independent controls use dimensions, candidate geometry and seed only, never relevance labels, questions, model predictions or sensitivities. Contextual irrelevance is not guaranteed.

| Operator | Requests | Mean mask area | Mean inside RGB change | Outside max | Sham-core max |
|---|---:|---:|---:|---:|---:|
| blur | 324 | 10.88% | 33.38 | 0 | 0 |
| lama | 324 | 10.88% | 35.56 | 0 | 0 |
| occlusion | 324 | 10.88% | 55.69 | 0 | 0 |

No identity removals/shams in this paired cohort. Exact masks, perimeters, inside/outside changes, SSIM maps, source image/annotation IDs and hashes are preserved. Whole-image SSIM can be dominated by unchanged pixels; local windows cross boundaries. Neither SSIM nor exact pixel restoration establishes semantic validity. Blur can retain coarse color; LaMa can reconstruct or invent content; rectangular overlays leak geometric cues; candidate controls need not match texture.

Official LaMa source786f5936b27fb3dacd2b1ad799e4de968ea697e7 and verified big-lama weights ran in the isolated CPU environment. No substitute editor. A fixed real-image official replay was byte-identical with0 pixel difference; CPU replay12.21s including loading. Batch logs, raw outputs and content caches remain local. The unscored original mixed-choice GQA edit cohort is preserved separately.

| Model | E2 score rows | Model interval(s) | Rows/s | NVML peakGiB | Torch allocated/reservedGiB | Peak hostRSS GiB |
|---|---:|---:|---:|---:|---|---:|
| internvl | 2025 | 289.58 | 6.99 | 7.34 | 5.23/6.15 | 5.23 |
| qwen | 2025 | 417.33 | 4.85 | 8.85 | 7.57/7.66 | 4.29 |
| smol | 2025 | 800.40 | 2.53 | 6.45 | 4.57/5.26 | 5.25 |

These measured intervals include model load, preprocessing, forwards and checkpoint I/O, not the prior integrity scan. NVML includes the Windows desktop and100ms sampling can miss short peaks. All models run sequentially in BF16, batch1, without quantization; LaMa never shares GPU memory. Per-row tensors, image-token counts and timings are in actual score records.

All6,075 paired score rows passed CPU verification of input/cache hashes, source/gold mapping, exact probability normalization, canonical answer labels, complete factorials and independent-control reconstruction. This is a computational pass only. The source-locomotive-box failure and other semantic uncertainties remain; see the dataset/failure reports.

**360 balanced randomized review packets**,15/task for each operator×candidate/control×removal/sham group, are at `outputs/v2/human_review_combined/`. Native-resolution lossless original/edited PNGs include exactly the scored task overlays; contact sheets add a cyan review-only extent pointer. Core and expanded masks are supplied. Reviewers must judge source referent matching and box completeness as well as removal, preservation, off-target validity, artifacts, unintended semantics and confidence. Keep the separate PRIVATE key hidden. Original initial-only packets also remain preserved. Zero submitted judgments; two real independent reviewers are still required. The ingestion tool rejects missing/duplicate raters, incomplete panels and missing ratings; uncertain judgments do not disappear from denominators. No ratings were fabricated.

E2 is **PARTIAL / EXPLORATORY**, never a semantic-validity pass. The fixed-grid diagnostic generated further edits for16 Qwen/InternVL images with49 scores/image; it is separately labeled no-oracle evidence candidates, not annotation-free task/image construction. E3's numerical sensitivity results remain provisional.
