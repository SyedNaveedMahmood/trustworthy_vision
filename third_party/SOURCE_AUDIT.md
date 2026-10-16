# Source and access audit — 2026-10-16

Machine records: [source_inventory.json](../reports/evidence/source_inventory.json). Exact fetched content is retained locally in ignored `outputs/sources/`; public records include response status, content SHA-256, byte count, retrieval errors and immutable revisions. A retrieved URL is not a reproduced implementation. No paper performance is asserted here.

| Resource | Pinned revision | Actual status |
|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | Public, ungated. **Qwen Research License, noncommercial research/evaluation**, not Apache. Weight download/smoke tracked separately. |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | `482adb537c021c86670beed01cd58990d01e72e4` | Public, ungated, Apache-2.0 model card. Weight download/smoke tracked separately. |
| Transformers | upstream HEAD `683b721971417b3b2c64fba9001f525607547e5a` | OFFICIAL_CODE, Apache-2.0. Execution uses pinned released **4.51.3**, not untested HEAD. Actual installed wheel lock is separate. |
| LaMa | `786f5936b27fb3dacd2b1ad799e4de968ea697e7` | OFFICIAL_CODE, Apache-2.0. Unmodified official CPU inference and identical-hash synthetic replay executed after dependency repairs. Natural-image semantic validity not audited. |
| REFER | `e3bbaa30d2ca41cf0e5c0d3819d7e4ed9fd38fff` | OFFICIAL_CODE, Apache-2.0 API. Its code license does not license every source photograph. Official schema used for debug input preparation. |
| TensorFlow Datasets | `848b71a1d322f476c3b6db103fcd0afe3ce8c9ce` | OFFICIAL_CODE, Apache-2.0. RefCOCO docs require manual preparation; UMD partitions images. |
| SAM (optional) | `dca509fe793f601edb92606367a655c15ac00fdf` | OFFICIAL_CODE, Apache-2.0. Not installed, loaded or evaluated. |
| CoEV | `39203e33a974c853aa0a8e77b6f7e7450a4e373b` | **UNAVAILABLE runnable code**: no Python files, only six-byte README. No source license found. No claimed replication. |

The runtime model wrappers use the official processors, chat templates and architecture implementations from Transformers. No custom VLM implementation or remote executable model code is used. Both primaries are frozen BF16 and must be loaded in separate processes. SHA-256 checksums of downloaded model files must be recorded after completion.

Executed Transformers release `v4.51.3` resolves to commit `5f4ecf2d9f867a1255131d2461d75793c0cf1db2`; see `reports/evidence/transformers_release_pin.json`. This differs from the separately audited upstream HEAD. Both primary weight snapshots and LaMa checkpoint checksums have now been recorded in `weight_checksums.json` and `lama_checkpoint.json`. A source hash or synthetic smoke does not reproduce a paper's benchmark performance.

## Data access

- [GQA official download/schema](https://cs.stanford.edu/people/dorarad/gqa/download.html): questions 1.2, scene graphs 1.1. Official archives support byte-range reads; debug preparation reads balanced validation questions and selected images with ZIP CRC validation, without downloading the complete image archive. The site's CC BY 4.0 notice is recorded; underlying photograph rights are not reassigned. No photographs or full annotations are committed.
- [REFER official README](https://github.com/lichengunc/refer): the original annotation host fails DNS. The [linked official issue](https://github.com/lichengunc/refer/issues/14) provides an archived copy. The documented Wayback snapshot was successfully downloaded using Windows TLS verification; CRC verified, 56,712,951 bytes, containing `refs(umd).p`, `refs(google).p`, `instances.json`. This resolves the initial annotation-access failure. Archive checksum is recorded with data preparation evidence.
- [TFDS official instructions](https://github.com/tensorflow/datasets/blob/master/docs/catalog/ref_coco.md): confirms manual source preparation and image-disjoint UMD definitions. No TFDS benchmark was run.
- COCO images: official HTTP distribution `http://images.cocodataset.org/train2014/`. The HTTPS virtual host currently presents a certificate-name mismatch; no verification was disabled. Per-image license IDs resolve through the archive's COCO `licenses` table. Raw images remain local.

## Closest baselines

| Method | Status for this execution | Reason |
|---|---|---|
| Raw occlusion, blur, mean/median, sham and off-target variants | PROXY / protocol baseline | Equations implemented as project controls; do not label them paper reproductions. Natural-image semantic validity remains unverified. |
| [EVICT](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Kotte_EVICT_Evidence-Sufficiency_Verification_via_Counterfactual_Dropout_for_Visually-Grounded_Selective_Question_CVPRW_2026_paper.html) | UNAVAILABLE verified runnable implementation | Official paper page fetched by local client. PDF/equation reproduction not yet completed. No invented baseline values. |
| [CoEV paper](https://papers.miccai.org/miccai-2026/0442-Paper0281.html) | INCOMPATIBLE full pipeline / UNAVAILABLE code | Medical claim extraction, radiology grounding and generated-text persistence are not four-choice region ranking. Project raw masks are not a faithful CoEV reproduction. |
| [CSGR](https://arxiv.org/abs/2609.13228) | UNAVAILABLE verified runnable implementation | Paper identified; initial Python certificate error repaired with native TLS fetch. No complete author implementation verified or run. |
| [VLM Reality Check](https://openaccess.thecvf.com/content/CVPR2026W/DataMFM/html/Sar_VLM_Reality_Check_A_Causal_Counterfactual_Benchmark_for_Diagnosing_Cognitive_CVPRW_2026_paper.html) | Optional benchmark, NOT RUN | Official page retrieved; no dataset substitution. |

No established method is assigned zero performance because access or compatibility is missing. Bibliographic details and DOI fields are not invented. Verified primary URLs above identify the sources; a full publication bibliography remains unfinished.
