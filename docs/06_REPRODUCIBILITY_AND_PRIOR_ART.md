# 06 — Verified reuse targets and limitations (checked 2026-10-16)

The coding agent must recheck URLs, releases, license terms, code availability, pinned commits, weights and dependencies. Do not claim a code release has been reproduced simply because a repository URL exists.

| Source | Verified item | Permitted role | Status |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct | Frozen primary VLM, official processor/model code via Transformers | Public model card/weights verified |
| SmolVLM2-2.2B-Instruct | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct | Frozen independently trained second VLM | Public model card/weights verified |
| Hugging Face Transformers | https://github.com/huggingface/transformers | Official model inference and logit scoring; pin version | Verified repo |
| LaMa (WACV 2022) | https://github.com/advimman/lama | Frozen edit operator; pin Git commit + model checkpoint hash | Verified runnable-source repository exists; compatibility not tested here |
| RefCOCO API | https://github.com/lichengunc/refer | Official referring expression parsing/evaluation utilities | Verified repo with code; dataset requires separate images/annotations |
| TensorFlow Datasets ref_coco | https://github.com/tensorflow/datasets/blob/master/docs/catalog/ref_coco.md | Split/access definitions and manual-download instructions | Verified official documentation |
| GQA | https://cs.stanford.edu/people/dorarad/gqa/download.html | Official question, semantic-program, scene-graph and box definitions | Verified official documentation |
| Segment Anything (SAM) | https://github.com/facebookresearch/segment-anything | OPTIONAL shared, label-free candidate region proposals; not needed in oracle-region pilot | Verified repo; weights/VRAM must be checked |
| CoEV (MICCAI 2026) | https://github.com/Zhounan1222/CoEV | Prior-method description/baseline if code appears | Verified repository currently ONLY a README containing "# CoEV"; DO NOT treat as runnable |
| EVICT (CVPRW 2026) | https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Kotte_EVICT_Evidence-Sufficiency_Verification_via_Counterfactual_Dropout_for_Visually-Grounded_Selective_Question_CVPRW_2026_paper.html | Faithful paper-described counterfactual dropout, if implementation available | No official code repository independently verified in drafting |
| CSGR (2026 preprint) | https://arxiv.org/abs/2609.13228 | Closest region-perturbation comparison | Full runnable release not independently confirmed |
| VLM Reality Check (CVPRW 2026) | https://openaccess.thecvf.com/content/CVPR2026W/DataMFM/html/Sar_VLM_Reality_Check_A_Causal_Counterfactual_Benchmark_for_Diagnosing_Cognitive_CVPRW_2026_paper.html | External counterfactual failure benchmark / optional stress test | Paper verified; dataset access/license should be checked |
| CoEV paper (MICCAI 2026) | https://papers.miccai.org/miccai-2026/0442-Paper0281.html | Paper-based specification and failure analysis | Paper and code URL verified; source repo stub |

## Official implementation policy

First try official code for each method, reading its README, training/inference assumptions, dependencies, license and exact output format. Create a reproduction test against authors' small published example if available. Do not install arbitrary dependencies into the main environment if it jeopardizes the VLM stack; isolate subprocess and container environments.

Never equate:
- A published article with an available GitHub implementation.
- A GitHub repository with an executable or license-clean implementation.
- A conceptually similar score with a faithful reproduced baseline.
- Running an author script with reproducing its reported benchmark result.

If a method has no usable official implementation, define the paper-equation reimplementation independently, unit-test it, state every deviation, and give it status PAPER_REIMPLEMENTATION. Proxy approaches belong under PROXY with explicit explanation. If the dataset/task cannot support a published baseline, mark INCOMPATIBLE, not zero performance. Do not fabricate numbers to make the baseline table complete.

## Expected source layout after implementation

- third_party/README.md: URLs, licenses, commit SHA, checkpoint SHA, integration flags and official-code availability.
- src/models/: small wrappers around official Transformers inference; no copied weight blobs.
- src/data/: versioned GQA and RefCOCOg loaders using official schemas.
- src/edits/: pinned LaMa adapter and image-equivalent operations.
- src/scoring/: fixed-choice log-probabilities, basic/dropout/sham/robust scores.
- src/evaluation/: paired metrics, stratified bootstrap, results validation.
- configs/: frozen YAML protocol and changes only through documented amendments.
- tests/: synthetic scoring, image-geometry, caching and leakage tests.
- outputs/: local ignored checkpoints/plots/results, small reports may be added to reports/.
- reports/: gate results and STOP/GO recommendation with honest missingness.

## Publication and citations

Use an APA-formatted bibliography with a DOI when verified, else a verified primary URL. Do not invent author lists, DOI, venue, accepted status or Git commits. The original research corpus comprises selected NeurIPS 2026 papers plus outside-conference counterexamples; do not claim it exhaustively covers CVPR, ICCV, ECCV, MICCAI or the 2026 preprint archive.

Dataset labels and paired interventions are imperfect proxies for semantics. Explicitly differentiate annotation-based relevance, observable sensitivity and inferred causal reliance.
