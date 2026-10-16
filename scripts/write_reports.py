"""Write explicit partial-run reports from saved observations, never missing estimates."""
import json
from pathlib import Path
from src.storage import atomic_json

R=Path("reports"); E=R/"evidence"
s=json.loads((E/"executed_summary.json").read_text())


def write(name,text):
    (R/name).write_text(text.strip()+"\n",encoding="utf-8")


table="| Task | Model | Clean | Text only | Shuffled image | Choice rotation | Clean Wilson 95% CI | Gate |\n|---|---|---:|---:|---:|---:|---|---|\n"
for task in ("gqa","refcocog"):
    for model in ("qwen","smol"):
        cell=s["cells"][f"{task}/{model}"]; m=cell["conditions"]
        values=[f"{m[c]['correct']}/40 ({m[c]['accuracy']:.1%})" for c in ("clean","text_only","shuffled_image","choice_rotation")]
        lo,hi=cell["accuracy_ci95"]["clean"]
        table+=f"| {task} | {model} | "+" | ".join(values)+f" | {lo:.1%}–{hi:.1%} | {'PASS' if m['competence_gate'] else 'FAIL'} |\n"

resources="| Model | Peak allocated GiB | Peak reserved GiB | NVML whole-device peak GiB | Peak host RSS GiB | E1 wall seconds |\n|---|---:|---:|---:|---:|---:|\n"
for model in ("qwen","smol"):
    t=json.loads((E/f"E1_{model}.json").read_text())["telemetry"]
    resources+=f"| {model} | {t['torch_peak_allocated_bytes']/2**30:.3f} | {t['torch_peak_reserved_bytes']/2**30:.3f} | {t['nvml_peak_device_used_bytes']/2**30:.3f} | {t['peak_process_rss_bytes']/2**30:.3f} | {t['wall_seconds']:.3f} |\n"

write("E0_SOURCE_AND_SMOKE.md",f"""
# E0: computational smoke completed; full gate not passed

2026-10-16, Asia/Dhaka. Source base `853f02959495f2056fd3ffea56ecf9b7e8bf3d27`; execution pins in `configs/runtime_pins.yaml`.

Both frozen models ran sequentially on CUDA BF16, batch 1, 512px maximum long edge, using official Transformers 4.51.3 processors/templates. Per model: 12 synthetic color-shape fixtures, 12 unique GQA images, 12 unique RefCOCOg images, and 4 occluded variants/task. Synthetic fixtures never enter benchmark manifests. Both scored 12/12 on synthetic color fixtures.

Every returned four-choice likelihood was finite. Token boundaries were checked by appending A/B/C/D to the exact rendered template; each label was one distinct token. The independent full-vocabulary log-softmax calculation matched conditional choice probabilities to floating-point precision. Repeated first-item logits were identical. See per-item `evidence/*_synthetic_smoke.json`; initial overlays remain in `evidence/*_overlay_v0_smoke.json`. The larger-letter repair did not fix SmolVLM2's 3/12 debug grounding result. This precludes a full E0 PASS. E1 was a flagged follow-up competence diagnosis, not a declaration that E0 passed.

Official LaMa source and author-linked weights were pinned and hashed. Its unmodified prediction script ran on an isolated CPU environment after protobuf and OpenCV repairs. Two successful executions returned identical image hashes (see `evidence/lama_replay.json`). Synthetic inpainting success is not semantic validity on natural images; no human gate passed.

GQA official balanced validation questions and scene graphs, RefCOCOg archived official UMD annotations, and selected official images were used. Debug and development byte hashes are disjoint. The official Visual Genome COCO-identity mapping found no known overlap among 104 sampled images; 25 GQA images have null COCO mappings, so identity coverage is incomplete. See `evidence/cross_dataset_identity_check.json`.

{resources}
NVML samples every 0.1s and includes the Windows desktop. Peak sampling can miss shorter transients; torch allocator peaks are exact for the instrumented processes. Physical capacity was measured as 16,376 MiB; all runs retained more than 1 GiB headroom. No quantization, OOM or failed model forward occurred.

27 core tests pass, including protocol YAML parsing, safe numerical logits, masks, resize geometry, label pairing, split leakage, cache tampering, pinned revisions, shared-image bootstrap, retry bounds, semantic-review flags and refusal to lock. Full result-row schema validation covered all 640 E1 rows. End-to-end E2–E5 implementations/tests remain unfinished.
""")

contrast=s["competence_macro_image_vs_text_contrast_not_H2"]
write("E1_COMPETENCE.md",f"""
# E1: two-model competence gate FAILED

The prescribed 40 unique development images/task were scored by both frozen models. No item was selected or dropped based on its prediction. The 80 development images are disjoint from the 24 debug images. Four conditions per item/model produced 640 score rows. Reference choices were frozen; option rotation changes the gold letter consistently while preserving the semantic answer. No-image prompts omit the image content item using the official chat template. Shuffled controls use the next image cyclically within task, resized to original dimensions, retaining the original candidate overlays.

{table}
The predeclared threshold is clean accuracy >=40% plus positive image-over-text accuracy gain for every model/task. SmolVLM2 RefCOCOg is 15/40 = 37.5%; it fails that deterministic engineering screen. Its Wilson interval crosses 40%, so this is not proof that population accuracy is below 40% or that the model fundamentally cannot ground expressions. The current configuration does not qualify for the prescribed full matrix.

The macro image-versus-text accuracy contrast is {contrast['effect']:+.2%}, with 95% paired cluster-bootstrap interval [{contrast['ci95'][0]:+.2%}, {contrast['ci95'][1]:+.2%}], 2,000 replicates, seed 20264. Images are resampled once within task and carried across both models. This is **not H2's controlled-estimator-versus-baseline effect**. Per-cell paired intervals, NLL, Brier scores, entropy, ten-bin descriptive ECE, subtype counts, permutation disagreement and timing are in `evidence/executed_summary.json`. Accuracy intervals in the figure/table are Wilson intervals, not bootstrap intervals.

Failure diagnosis: SmolVLM2 predicts B on 20/40 clean grounding items and 32/40 shuffled grounding items. Choice rotation reduces grounding accuracy to 13/40. Both GQA models have strong text-only performance (72.5% and 77.5%); the automatically constructed four-choice subset may be easy or prior-driven. These results do not validate full GQA or free-form VQA competence. GQA development sampling used one explicit referenced object and a same-global-answer-group distractor pool; broad relation/spatial strata and exhaustive distractor ambiguity audits were not completed. Candidate box overlap/texture eligibility for later region interventions is also unfinished. No post-outcome exclusion or replacement was made.

The independent-family fallback checked was `google/paligemma2-3b-mix-448`, revision `1406c92ec87d32cc6b983239278901b904ba7a51`. The official API reports manual gating and its config returns HTTP 401. It was not loaded or fit-tested; no assertion that all possible other backbones are unavailable is made. See `evidence/fallback_access.json`.

![Executed development competence](figures/E1_competence.png)
""")

blocked={
"E2_EDITOR_VALIDITY.md":("E2: NOT EXECUTED — prerequisite failed","Official LaMa synthetic replay now works, and occlusion/blur geometry has automated tests. The full 80-image removal/sham/off-target factorial, blinded natural-image contact sheets and two-rater semantic review were not executed. Removal success, sham preservation and off-target validity are NOT MEASURED; do not replace them with zero or the target thresholds. HUMAN_AUDIT_PENDING. No natural-image operator panel is semantically validated."),
"E3_DEV_BASELINES.md":("E3: NOT EXECUTED — no eligible full matrix","The exact median-minus-MAD equation and control contrasts have numerical unit tests, but no development region-pair baseline comparison was run. No strongest comparator or lambda was selected. Raw/sham/operator variants in code are project controls, not faithful EVICT/CoEV/CSGR reproductions. Source availability and compatibility are recorded in third_party/SOURCE_AUDIT.md. The 13-image design's oracle-negative reuse conflicts with the label-free control requirement; mutual controls cancel the MAD penalty in the pair contrast. This needs a documented control construction before any method claim."),
"E4_ABLATIONS.md":("E4: NOT EXECUTED — E2/E3 prerequisites absent","No leave-one-operator-out, held-out severity/fill, dose-response, estimator mean/median or dispersion ablation was scored. E1 option rotation is a competence diagnostic, not a completed E4 robustness study. The algebraic mutual-control cancellation test is a protocol diagnostic, not evidence of empirical operator stability."),
"E5_CONFIRMATORY.md":("E5: BLOCKED — protocol NOT LOCKED","Zero confirmatory images were sampled and zero confirmatory forwards were made. No comparator, lambda or thresholds were frozen from development region-pair results. There is no primary effect, CI, p-value, AUROC or selective-risk result for the controlled estimator. No post-hoc tuning or test-set access occurred. Resolving failed model/task competence, operator validity and label-free control construction is required before a genuine lock. A shared independent extra control would expand 13 to 19 images/item: 3,040 development plus 9,120 confirmatory forwards already exceed the 12,000-call total budget before competence/ablations; this variant requires a scoped pre-confirmatory amendment rather than a silent expansion.")}
for filename,(heading,body) in blocked.items():
    write(filename,f"# {heading}\n\nThe required model/task feasibility screen failed: SmolVLM2 RefCOCOg 15/40 < 40%. No eligible replacement was executed.\n\n{body}\n\nPreserved evidence: `evidence/E1_smol.json`, `evidence/fallback_access.json`, `DEVIATIONS.md`. These are blocker records, not completed experiment result tables.")

write("FINAL_VERDICT.md",f"""
# INCONCLUSIVE

The prescribed controlled-evidence hypothesis has not been tested. GPU execution and development competence diagnostics ran, but the required model/task matrix failed its feasibility screen and no eligible independent-family replacement was executed. No scientific GO or estimator NO-GO can be supported from these observations.

{table}
Baseline comparisons actually run are clean versus no-image, shuffled-image and choice rotation. The macro clean-minus-no-image accuracy contrast is {contrast['effect']:+.2%}, 95% image-cluster bootstrap [{contrast['ci95'][0]:+.2%}, {contrast['ci95'][1]:+.2%}]. These controls show some image sensitivity alongside strong language priors and grounding label bias. They do not establish semantic-evidence dependence or improvement over a strong region-sensitivity baseline.

{resources}
Actual model forwards: **{s['resources']['forward_calls']}**, failed forwards: **{s['resources']['failed_forward_calls']}**. Synchronized forward wall time: **{s['resources']['synchronized_forward_wall_seconds']:.3f}s ({s['resources']['synchronized_forward_wall_hours']:.5f}h)**. Summed instrumented model-process wall time, including loading/preprocessing and failed processor setup: **{s['resources']['instrumented_model_process_wall_seconds']:.3f}s ({s['resources']['instrumented_model_process_wall_hours']:.5f}h)**. Active GPU kernel hours were not profiled. No 24-hour or 12,000-forward ceiling was reached. LaMa ran on CPU and has no claimed GPU resource measurement.

Completed work: source/model/checkpoint hashes; frozen sequential BF16 device runs; 12 debug and 40 development images/task shared by models; 640 development condition scores; 27 core tests; schema/numerical validation; bootstrap competence intervals; official LaMa synthetic inference and replay; recovered dependency/network failures; code, logs and local checkpoints. Initial failed rendering and setup records remain preserved.

Unfinished: complete E0 semantic/editor gate, qualified second-family grounding configuration, E2 natural-image validity audit, E3 region-pair baseline selection/comparison, E4 operator robustness, G5 immutable confirmatory lock, E5 120-image/task confirmation and H2 uncertainty analysis. E6 is this honest partial-run/failure audit, not a completed full-study review. No human ratings or primary-method results were fabricated.

Next required changes: obtain legitimate fallback model access or preregister another independent model; rerun qualification on a separately declared development procedure without using confirmatory outputs; complete two-rater edit review; resolve/cost label-free controls and the budget conflict; finish later-stage orchestration and tests; freeze code/config/prompts/comparator plus an image-disjoint confirmatory manifest before E5. No work continues in the background.

Exact commands and preserved artifact locations are in REPRODUCIBILITY.md. Source references are in third_party/SOURCE_AUDIT.md. All claims are conditional on the small automatically constructed development subset; uncertainty intervals do not repair annotation, distractor or semantic-validity limitations.
""")
print("Wrote E0–E5 status reports and partial final verdict from executed evidence")
