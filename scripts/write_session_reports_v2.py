"""Generate the scientific handoff from completed, verified experiment artifacts."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
from src.session import git_revision


def load(name):return json.loads(Path('reports/evidence/v2/'+name+'.json').read_text())
def pct(x):return f'{100*x:.2f}%'
def interval(values):return '['+', '.join(f'{100*x:+.2f}' for x in values)+'] pp'


def run():
    comparison=load('E3_replication');macro=load('E3_pooled_macro_summary');resources=load('resource_summary');quality=load('edit_quality_and_human_review_combined');verified=load('artifact_verification');qualification=load('corrected_qualification_summary')
    assert verified['verified_score_rows']==6075 and verified['status']=='PASS_CPU_ARTIFACT_VERIFICATION'
    assert resources['budget']['forward_calls']<=12000
    selected=comparison['comparator_proposal'];effect=selected['candidate_vs_selected_baseline'];budget=resources['budget'];commit=git_revision();now=datetime.now(timezone.utc).isoformat()
    tests=ET.parse('reports/evidence/v2/tests_v2.xml').getroot();test_n=sum(int(s.get('tests','0')) for s in tests.findall('testsuite'))
    table='| Method | Macro pair ranking (image-cluster95% CI) | Direct scores/pair |\n|---|---|---:|\n'
    costs={'raw_occlusion':3,'raw_mean':7,'raw_median':7,'sham_median':13,'robust_sham':13,'robust_independent':25}
    for method,stats in macro['methods'].items():table+=f"| {method} | {pct(stats['effect'])} [{pct(stats['ci95'][0])}, {pct(stats['ci95'][1])}] | {costs[method]} |\n"
    cells='| Task | Model | n | Raw occlusion | Raw median | Sham median (S1) | Robust sham (S2) | Independent robust (S3) |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for task in ['gqa','refcocog']:
        for model in ['qwen','smol','internvl']:
            cell=comparison['cells'][f'pooled/{task}/{model}/gold'];cells+=f"| {task} | {model} | {cell['raw_median']['n']} | "+' | '.join(pct(cell[m]['ranking_accuracy']) for m in ['raw_occlusion','raw_median','sham_median','robust_sham','robust_independent'])+' |\n'
    qual='| Model | Color GQA image/text/shuffled (/20) | RefCOCOg image/text (/20) | Screen |\n|---|---|---|---|\n'
    for model in ['qwen','smol','internvl']:
        q=qualification['models'][model];a=q['gqa']['metrics'];b=q['refcocog']['metrics'];qual+=f"| {model} | {a['image']}/{a['text_only']}/{a['shuffled']} | {b['image']}/{b['text_only']} | {a['passed'] and b['passed']} |\n"
    hardware='| Model | E2 score rows | Model interval(s) | Rows/s | NVML peakGiB | Torch allocated/reservedGiB | Peak hostRSS GiB |\n|---|---:|---:|---:|---:|---|---:|\n'
    for model,r in resources['paired_models'].items():hardware+=f"| {model} | {r['paired_score_rows']} | {r['instrumented_model_interval_seconds']:.2f} | {r['score_rows_per_instrumented_second']:.2f} | {r['nvml_peak_device_used_bytes']/2**30:.2f} | {r['torch_peak_allocated_bytes']/2**30:.2f}/{r['torch_peak_reserved_bytes']/2**30:.2f} | {r['peak_host_rss_bytes']/2**30:.2f} |\n"
    matrix='''| Stage | Status | Actual execution / validation limit |
|---|---|---|
| E0 | PARTIAL / EXPLORATORY | Three official frozen backbones fit BF16; processor/likelihood/repeat tests and real official LaMa replay completed. Independent semantic editor validity remains pending. |
| E1 | COMPLETE (amended engineering screen) | Fresh20 images/task/model, fixed40% screen and positive image gain. Original broad GQA competence is unvalidated; replacement is explicitly color-only. |
| E2 | PARTIAL / EXPLORATORY | 81 paired images shared by3 backbones;6,075 complete score rows. 16 grid images/backbone for Qwen/InternVL. Pixel/hash checks pass; no semantic audit pass. |
| E3 | PARTIAL / EXPLORATORY | Actual baselines, S1–S3, initial/extension/pooled comparisons, paired bootstrap, AUROC and cost analysis. All development; no confirmatory claim. |
| E4 | PARTIAL / EXPLORATORY | Real resolution/letter/fill/blur/mask studies for Qwen/InternVL; matched five/task corrected-Smol dose study; numerical ablations all3 backbones. Not every factor/model/operator transfer completed. |
| E5 | BLOCKED | Zero confirmatory manifests or forwards. Human/control/reference validity, genuine lock and feasible amended sample/budget remain absent. |
| E6 | PARTIAL / EXPLORATORY | Current negative-result/failure audit, verdict and handoff complete; full-study interpretation remains contingent on E5. |
'''
    caveats='''These are annotation-reference region rankings, not verified causal relevance. Source annotations can be incomplete: GQA1593212 asks locomotive color but its official box[875,671,977,761] covers the lower coupler in a1024×768 image, leaving most of the train visible. Agent visual inspection is not an independent human judgment. This case and all other failures remain included. Confidence intervals do not repair this label problem, semantic edit errors, cross-dataset identities missing from available mappings, or adaptive development reuse.

The baseline was chosen on the same pooled development results. Its paired bootstrap interval is descriptive and does not account for comparator selection. No confirmatory p-value, positive effect claim, equivalence claim or statistically validated deployment reliability is asserted. Three operators are correlated, and their scaled MAD is a disagreement penalty, not a standard error. The original practical development target is+3pp; current S3 has not supported it. No coefficient or threshold was changed to rescue the outcome.
'''
    result=f'''# E3 exploratory comparison — completed development replication

**S3 has not demonstrated added value.** On the intended Qwen + corrected-Smol pair, S3 ranks {pct(macro['methods']['robust_independent']['effect'])} versus {pct(macro['methods']['raw_median']['effect'])} for the development-selected raw-median baseline: **{100*effect['effect']:+.2f} percentage points**, descriptive paired95% CI **{interval(effect['ci95'])}**. Direct cost is25 versus7 model-image scores/pair. Overall scientific verdict: **INCONCLUSIVE**; do not launch confirmation of this formulation.

The initial15 GQA/16 RefCOCOg pilot and independent fixed25/task extension are preserved separately. Pooled40 color GQA/41 RefCOCOg images are shared across all3 backbones (81 independent image clusters, not243 independent model observations). Gold and fixed-original-prediction anchors are separate. No outcome-dependent image filtering. The extension was declared after initial results, with fixed formulas/operators/eligibility; it is further development, never confirmation.

{table}
![Pooled primary comparison](evidence/v2/E3_pooled_primary.png)

{cells}
![Paired contrasts](evidence/v2/E3_pooled_contrasts.png)

![Inference cost](evidence/v2/E3_cost_accuracy.png)

Raw median is the single **DEVELOPMENT_SELECTED_NOT_LOCKED** comparator proposal. Selection maximizes equal-weight task/model gold-anchor ranking among the declared baselines, then minimizes direct calls and uses lexical tie break. S2 is competitive in aggregate but does not establish a meaningful gain: raw median85.53%, S2 85.76%, S3 80.23%. Candidate-specific control subtraction is harmful relative to S2 on these data: paired effect−5.53pp, descriptive95% CI[−10.44,−1.22]. This is not a reason to tune S3 post hoc.

Pair ranking gives half credit to ties. Within-pair AUROC equals this concordance; pooled-region AUROC is separately reported. Bootstrap resamples image IDs once per task, carrying both primary models jointly;5,000 replicates, seed20269. Cell summaries use2,000 replicates with the same image-pair alignment. Supplementary Wilson bounds are provided for tie-free cells; small-sample empirical intervals can be degenerate. No multiplicity-adjusted confirmatory inference is made.

Full tables, frozen-prediction anchors, per-cell intervals, negative-score exceedances, ablations and initial/extension estimates: [replication tables](evidence/v2/E3_replication_tables.md), [complete analysis](evidence/v2/E3_replication.json). Earlier Qwen/InternVL-only results remain in `E3_comparison.json`; corrected initial Qwen/Smol results are `E3_corrected_initial.json`.

Confidence and entropy are image-level selectors, not region-ranking baselines; a shared clean confidence cannot distinguish two regions. Clean calibration and oracle-region selection are in `clean_calibration_and_oracle_selectivity.json`. S3's predicted-answer, oracle-region AURC versus confidence differs+0.0183[−0.0241,+0.0620], where positive is worse. This does not improve deployment reliability. The separate fixed-grid no-oracle evidence diagnostic remains limited to Qwen/InternVL, eight images/task/model; S3 loses to confidence in3/4 cells at49 versus1 scores/image. Smol grid inference was cancelled to preserve the cumulative cap and prioritize correcting its adapter.

S1=median sham-corrected effects; S2=S1−0.5 scaledMAD; S3 applies independent candidate-control subtraction before median/MAD. Gold anchors are oracle-answer diagnostics. Predicted anchors freeze the original unedited answer and never follow edited argmax; annotated candidate pairs remain oracle-region diagnostics. Negative>1-logit exceedances are not validated causal false-positive rates.

{caveats}
Source audit: `BASELINE_SOURCE_AUDIT_V2.md`. EVICT equations2–3 were tested as a gray128, semantic-option-text, fixed-choice adapter; no full official reproduction is claimed. CSGR/CoEV full systems remain unavailable/incompatible in the audited scope. Official LaMa and all3 model checkpoints were actually executed at pinned revisions.
'''
    Path('reports/E3_EXPLORATORY_COMPARISON.md').write_text(result,encoding='utf-8')
    geometry=quality['geometry'];gtable='| Operator | Requests | Mean mask area | Mean inside RGB change | Outside max | Sham-core max |\n|---|---:|---:|---:|---:|---:|\n'
    for op,r in geometry['by_operator'].items():gtable+=f"| {op} | {r['n']} | {pct(r['mean_expanded_fraction'])} | {r['mean_inside_absolute_change']:.2f} | {r['outside_mask_max_change']} | {r['sham_core_max_change']} |\n"
    e2=f'''# E2 exploratory results — completed computational development

**81 real images, three qualified frozen backbones,6,075 complete paired score rows; HUMAN_AUDIT_PENDING.** Initial15 GQA/16 RefCOCOg plus25 new images/task. Every image has clean+24 edits per model. All3 backbones use identical cached edited images; no confirmatory sample was accessed.

{geometry['edit_requests']} content/mask/operator requests produce {geometry['rendered_images']} removal/sham images. Mean-fill occlusion, Gaussian blur sigma24 and official LaMa share canonical RGB capped at768 and a fixed8px halo. Shams restore exact original core pixels; outside-mask compositing is exact. Independent controls use dimensions, candidate geometry and seed only, never relevance labels, questions, model predictions or sensitivities. Contextual irrelevance is not guaranteed.

{gtable}
No identity removals/shams in this paired cohort. Exact masks, perimeters, inside/outside changes, SSIM maps, source image/annotation IDs and hashes are preserved. Whole-image SSIM can be dominated by unchanged pixels; local windows cross boundaries. Neither SSIM nor exact pixel restoration establishes semantic validity. Blur can retain coarse color; LaMa can reconstruct or invent content; rectangular overlays leak geometric cues; candidate controls need not match texture.

Official LaMa source786f5936b27fb3dacd2b1ad799e4de968ea697e7 and verified big-lama weights ran in the isolated CPU environment. No substitute editor. A fixed real-image official replay was byte-identical with0 pixel difference; CPU replay12.21s including loading. Batch logs, raw outputs and content caches remain local. The unscored original mixed-choice GQA edit cohort is preserved separately.

{hardware}
These measured intervals include model load, preprocessing, forwards and checkpoint I/O, not the prior integrity scan. NVML includes the Windows desktop and100ms sampling can miss short peaks. All models run sequentially in BF16, batch1, without quantization; LaMa never shares GPU memory. Per-row tensors, image-token counts and timings are in actual score records.

All6,075 paired score rows passed CPU verification of input/cache hashes, source/gold mapping, exact probability normalization, canonical answer labels, complete factorials and independent-control reconstruction. This is a computational pass only. The source-locomotive-box failure and other semantic uncertainties remain; see the dataset/failure reports.

**360 balanced randomized review packets**,15/task for each operator×candidate/control×removal/sham group, are at `outputs/v2/human_review_combined/`. Native-resolution lossless original/edited PNGs include exactly the scored task overlays; contact sheets add a cyan review-only extent pointer. Core and expanded masks are supplied. Reviewers must judge source referent matching and box completeness as well as removal, preservation, off-target validity, artifacts, unintended semantics and confidence. Keep the separate PRIVATE key hidden. Original initial-only packets also remain preserved. Zero submitted judgments; two real independent reviewers are still required. The ingestion tool rejects missing/duplicate raters, incomplete panels and missing ratings; uncertain judgments do not disappear from denominators. No ratings were fabricated.

E2 is **PARTIAL / EXPLORATORY**, never a semantic-validity pass. The fixed-grid diagnostic generated further edits for16 Qwen/InternVL images with49 scores/image; it is separately labeled no-oracle evidence candidates, not annotation-free task/image construction. E3's numerical sensitivity results remain provisional.
'''
    Path('reports/E2_EXPLORATORY_RESULTS.md').write_text(e2,encoding='utf-8')
    verdict=f'''# INCONCLUSIVE

The corrected evidence estimator is **not supported as an improvement by the executed development study**. Do not advance S3 to confirmation on these results. The full scientific hypothesis remains inconclusive because intervention/reference validity is not independently established and E5 has not run.

The intended Qwen + corrected-Smol pair scores S3 {pct(macro['methods']['robust_independent']['effect'])} versus raw median {pct(macro['methods']['raw_median']['effect'])}: {100*effect['effect']:+.2f}pp, descriptive paired95% CI{interval(effect['ci95'])}. S3 costs25 versus7 scores/pair. The +3pp practical development target was not supported. Comparator selection used these same development data; its interval is descriptive, not confirmatory or selection-adjusted. We did not retune the coefficient to force a positive result.

{table}
{cells}
**The model feasibility problem was an adapter bug, not demonstrated incapability.** Smol expects space-prefixed letter tokens; the old bare-label tokens received only~4.5e-6 combined vocabulary mass. Corrected labels receive~.9998. All historical outputs remain archived and are invalidated as Smol capability estimates. Fresh disjoint qualification gives:

{qual}
Original GQA distractors also mixed answer types on18/40 development questions. The replacement color-only cohort is a separate narrower experiment; broad GQA reasoning is unvalidated. A concrete source bounding-box completeness failure remains unfiltered. No causal evidence claim follows from a clean numerical score or bootstrap.

{matrix}
{hardware}
Actual cumulative reserved model-forward attempts: **{budget['forward_calls']}/12,000**, failed model calls **{budget['failed_calls']}**, remaining **{resources['remaining_forward_allowance']}**. This session added {resources['new_session_calls']} calls to the preserved776 prior calls. Synchronized forward wall time{budget['forward_wall_seconds']:.2f}s. CUDA stream elapsed{budget['cuda_stream_seconds']:.2f}s covers{resources['cuda_stream_timed_call_count']} calls after call{resources['cuda_stream_timing_starts_after_call']}; it is not summed active-kernel time. Conservative GPU-time upper bound at the last call{budget['gpu_time_conservative_upper_bound_seconds']/3600:.3f}h remains below24h. Device-utilization/power/temperature observations are in `device_observer_summary.json`; they cover only their actual observation window and include desktop activity.

There are{test_n} passing tests and6,075 cache-verified paired score rows. A12-call official last-logit optimization probe preserved choice logits on six inputs but changed some non-choice logits by up to.03125 and provided no meaningful speed/memory gain; it was not applied to main results. GPU inference is finished; no model process is left to continue this study automatically.

**E5 launch: NO-GO.** Zero confirmatory manifests/forwards. Independent human review remains HUMAN_AUDIT_PENDING; original120/task at25 scores×2 models costs12,000 new calls alone, exceeding the cumulative cap. Remaining16 calls cannot fund even one25-score pair. No budget reset, sample reduction or protocol lock was silently performed. The next valid study requires source/edit review and an explicitly amended feasible design; current raw-median comparator proposal is development-selected, not locked.

{caveats}
Full execution, preserved artifacts, remaining dependencies and GitHub links: [handoff](TWO_HOUR_HANDOFF.md), [E2](E2_EXPLORATORY_RESULTS.md), [E3](E3_EXPLORATORY_COMPARISON.md), [E4](E4_EXPLORATORY_ABLATIONS.md), [reproduction](../REPRODUCIBILITY.md). Earlier verdict is preserved verbatim in `FINAL_VERDICT_V1_ARCHIVE.md`.
'''
    path=Path('reports/FINAL_VERDICT.md');archive=Path('reports/FINAL_VERDICT_V1_ARCHIVE.md')
    if not archive.exists():shutil.copy2(path,archive)
    path.write_text(verdict,encoding='utf-8')
    handoff=f'''# Two-hour research handoff

Generated{now}. Original autonomous window07:18:11–09:18:11UTC on2026-10-16. GPU execution completed around08:40UTC. The user later said “continue”; remaining CPU validation/reporting resumed after the original window. No GPU inference continued during the gap or after its completed runs. This file does not claim two hours of continuous GPU work.

## A. Executive summary

**Scientific verdict: INCONCLUSIVE. E5 launch: NO-GO.** Implemented and executed corrected model qualification, three intervention operators with shams/independent controls, real E2/E3 development comparisons, E4 ablations, no-oracle evidence diagnostics, statistical analysis, integrity checks and review material. S3 loses to the strong simple comparator in point estimate and consumes more inference. Human semantic review is still absent; no confirmation was attempted.

S3 versus raw median: {100*effect['effect']:+.2f}pp, descriptive paired95% CI{interval(effect['ci95'])}; {pct(macro['methods']['robust_independent']['effect'])} versus {pct(macro['methods']['raw_median']['effect'])}. 81 image pairs shared across3 backbones;6,075 scored cells. All historical negative outcomes are preserved. {test_n} tests pass.

## B. Model qualification

{qual}
Fresh20 images/task are disjoint from earlier data and extension source images. Default768/legacy fixed before these outputs. Intended Qwen+Smol pair restored; InternVL remains a useful third-family comparison. Full-vocabulary audit identified Smol's missing leading-space continuation: prior single-token tests checked the wrong canonical surface form. Corrected development512/768 scores19/40 and18/40; numbered19/40, outer rotation15/40, visual relabeling18/40. Source-native Ref images<=640px mean1024 adds no detail beyond768. Old raw-token resolution/presentation outputs remain archived; they are not valid capability conclusions. Official source/processor revisions, licenses, dependency locks, image-token counts and paired image gains are in `MODEL_QUALIFICATION_V2.md` and runtime_v2.yaml.

## C. Method correction

Mutual positive/negative controls force z−=−z+, equalMAD and exact cancellation of the dispersion penalty in pair ranking. Numerical/property tests demonstrate it. Amendments08–10 preserve the original protocol while defining S1=median(sham−removal conditional-choice log odds), S2=S1−.5MAD, and S3=median(candidate−independent-control effects)−.5MAD. Controls accept only candidate geometry, image dimensions and seed. Oracle gold and frozen-original-prediction anchors are separate; annotation-selected region pairs remain oracle-region diagnostics. A fixed-grid candidate diagnostic was executed separately for Qwen/InternVL. Lambda remains.5. The additional control has not justified its doubled edited-image cost.

## D. Experiment completion matrix

{matrix}
Implementation, actual execution and scientific validation are intentionally separate. No all-stage pass or full E6 conclusion is claimed.

## E. Quantitative evidence and resources

{table}
{cells}
{hardware}
Budget: {budget['forward_calls']} cumulative attempts, {budget['failed_calls']} failed calls, {resources['remaining_forward_allowance']} remaining;24h ceiling respected. Forward wall{budget['forward_wall_seconds']:.2f}s; CUDA stream elapsed{budget['cuda_stream_seconds']:.2f}s on{resources['cuda_stream_timed_call_count']} instrumented calls; conservative upper bound at last call{budget['gpu_time_conservative_upper_bound_seconds']/3600:.3f}h. GPU events are not active-kernel hours. Official LaMa used its isolated CPU stack and reproduced a real image byte-for-byte. All paired outputs pass input/cache/mapping/probability/control checks. Operator edits preserve outside pixels and sham cores exactly; these are not semantic validity rates.972 edit requests yield1,944 paired removal/sham images. Zero human-review judgments or semantic success-rate estimates exist.

## F. Scientific conclusions

The original model feasibility concern is repaired, and visually dependent color/grounding diagnostics run on three fitting backbones. Broad GQA task validity is not established. S3 is not promising in its current expensive form:80.23% versus85.53% for raw median. S2 reaches85.76%, but does not establish a meaningful improvement at additional cost. Source boxes, semantic edits and controls remain unvalidated. Oracle-region sensitivity does not demonstrate automatic localization or deployment reliability. The tiny grid study is worse than confidence in3/4 cells; its n=8 limits inference. No positive result, human judgment or official baseline reproduction was invented.

## G. Remaining blockers

Manual/scientific: two real independent reviewers must inspect the blinded native-resolution packets, validate question/box completeness, removal>=90%, sham preservation>=95%, and off-target validity>=95%. Distinct declared IDs alone do not authenticate independence. The source locomotive/coupler case must be adjudicated; current rows must remain in the original accounting. Human review is HUMAN_AUDIT_PENDING.

Compute/protocol: only16 of12,000 calls remain. The unchanged120/task independent-control confirmation needs12,000 additional calls. E5 cannot proceed by resetting counters, relaxing screens or silently reducing sample size. A new feasible, explicitly amended study decision is required. PaliGemma authorization is not an active blocker: an ungated independent fallback was actually run.

Engineering left for a future authorized study: full Smol grid/operator-transfer coverage, all-factor/model ablations, a fully compatible CSGR/CoEV reproduction if code becomes usable, an externally validated proposal/evidence reference, genuine parameter/comparator/manifest lock, and actual E5/E6 completion. These were not substituted with fabricated or proxy results.

## H. Exact next steps and commands

1. Keep the PRIVATE key hidden. Two reviewers independently copy `outputs/v2/human_review_combined/review_form.csv` to `review_form_reviewer1.csv` and `review_form_reviewer2.csv`, fill real IDs/ratings, and inspect `full_resolution/` lossless pairs plus masks. Run `.venv-core\\Scripts\\python.exe -m scripts.ingest_human_review_v2`. No reviewer rows are currently submitted.
2. Audit source-region completeness and contextual controls; decide a future task/operator amendment before any confirmation. Do not filter current failures retrospectively.
3. Retain raw_median as the development comparator proposal. Decide whether the current S3 should be abandoned or a separately motivated version studied; do not tune this run to claim success.
4. Resolve compute/sample feasibility without exceeding the existing cap. `prepare_confirmatory_v2` writes only a blocked snapshot; no manifest has been generated.
5. Reanalyze preserved results without GPU: commands below. Inference commands enforce the original active window and cumulative ledger; future inference requires an authorized, versioned execution decision.

```powershell
.venv-core\\Scripts\\python.exe -m scripts.analyze_replication_v2
.venv-core\\Scripts\\python.exe -m scripts.plot_replication_v2
.venv-core\\Scripts\\python.exe -m scripts.analyze_clean_predictions_v2
.venv-core\\Scripts\\python.exe -m scripts.analyze_ablations_v2
.venv-core\\Scripts\\python.exe -m scripts.analyze_matched_dose_v2
.venv-core\\Scripts\\python.exe -m scripts.verify_experiment_artifacts_v2 --require-complete
.venv-core\\Scripts\\python.exe -m scripts.prepare_confirmatory_v2
.venv\\Scripts\\python.exe -m pytest -q
```

Artifact verification and dose reconstruction require ignored local manifests/caches; score reanalysis/plots require only published JSON and analysis dependencies. Stage CLI: `src.cli run --config configs/exploratory_v2.yaml --stage E2/E3/E4`; use one stage value, not the slash expression. E4 `--model smol` uses the bounded five/task dose study. Source/weights are in local HF caches, third_party_weights and third_party/repos; selected datasets in datasets; atomic row checkpoints and edited images in outputs/v2 and outputs/cache. No weights or raw dataset images are pushed. [Reproduction details](../REPRODUCIBILITY.md).

## I. GitHub and modified files

Branch: [pilot/controlled-evidence-v1](https://github.com/SyedNaveedMahmood/trustworthy_vision/tree/pilot/controlled-evidence-v1). [Latest branch commits](https://github.com/SyedNaveedMahmood/trustworthy_vision/commits/pilot/controlled-evidence-v1). Evidence-parent commit when this report was generated: [{commit}](https://github.com/SyedNaveedMahmood/trustworthy_vision/commit/{commit}); report publication follows on the same branch. Use `git rev-parse HEAD` for the checked-out final handoff commit; no self-referential commit hash is fabricated.

Main files: [qualification](MODEL_QUALIFICATION_V2.md), [mathematical audit](ESTIMATOR_MATHEMATICAL_AUDIT.md), [dataset/prompt audit](DATASET_AND_PROMPT_AUDIT.md), [E2](E2_EXPLORATORY_RESULTS.md), [E3](E3_EXPLORATORY_COMPARISON.md), [E4](E4_EXPLORATORY_ABLATIONS.md), [verdict](FINAL_VERDICT.md), [status](STATUS.md), [deviations](DEVIATIONS.md); `docs/08–10`, configs/runtime_v2.yaml, requirements locks, src/models/choice.py, src/models/internvl.py, src/scoring/v2.py, src/edits/pilot.py, src/evaluation/, scripts/*v2.py and tests/test_v2*.py. Public metrics/figures/checksum evidence: `reports/evidence/v2/`. Original E0/E1 JSON is unchanged. No main merge, force push, reset, overwritten user work or fabricated human audit.
'''
    Path('reports/TWO_HOUR_HANDOFF.md').write_text(handoff.replace(chr(92)*2,chr(92)),encoding='utf-8')
    status=f'''# Execution status — amended development study

Updated{now}; evidence-parent commit `{commit}`, branch `pilot/controlled-evidence-v1`.

**INCONCLUSIVE scientifically; E5 launch NO-GO.** All planned permissible GPU execution is finished at{budget['forward_calls']}/12,000 calls, with16 remaining and0 failed model calls. No GPU experiment is continuing in the background. User resumed CPU reporting after the original autonomous window; no inference ran during that gap.

{matrix}
Fresh qualification: Qwen color13/20 versus9/20 text, Ref15/20 versus4/20; corrected Smol color12/20 versus7/20, Ref13/20 versus4/20; InternVL color13/20 versus7/20, Ref12/20 versus5/20. Intended Qwen+Smol pair restored. The earlier Smol failure was a canonical continuation error; old outputs remain preserved, not capability evidence.

81 development image pairs×3 models×25 scores=6,075 verified rows. S3 80.23% versus raw median85.53%, paired difference−5.30pp, descriptive95% CI[−12.02,+1.14]. Raw median is proposed from development only, not confirmatory locked. {test_n} tests pass. Actual blur/mask/fill/resolution/permutation ablations, frozen-answer analyses, no-oracle grid diagnostics, real official LaMa replay and full resource measurements are preserved.

360 combined blinded packets with native-resolution scored overlays are ready; **HUMAN_AUDIT_PENDING**, zero judgments. Source-box completeness and edit/control semantics remain unresolved. Original GQA choices mixed types; the new color task is narrower. Original120/task confirmation alone needs12,000 additional calls under the repaired factorial, so it cannot fit the remaining budget. No confirmatory manifest, inference or lock exists.

See [complete handoff](TWO_HOUR_HANDOFF.md), [verdict](FINAL_VERDICT.md), per-stage reports and `reports/evidence/v2/execution_ledger.json`. Ignored local datasets, edited images, model weights and atomic checkpoints remain on disk; small metrics/reports/configurations/code are pushed to the pilot branch. Prior status/verdict and original E0/E1 observations remain available in Git history and FINAL_VERDICT_V1_ARCHIVE.md.
'''
    Path('reports/STATUS.md').write_text(status,encoding='utf-8')
    readme=Path('README.md').read_text(encoding='utf-8');lines=readme.splitlines();lines[2]='Status: AMENDED DEVELOPMENT EXPERIMENTS EXECUTED (2026-10-16). Three frozen BF16 backbones,81 paired images,6,075 verified intervention scores; corrected Smol qualifies. S3 underperforms raw median in development. Verdict: INCONCLUSIVE; E5 blocked by independent semantic review and cumulative compute. See [handoff](reports/TWO_HOUR_HANDOFF.md), [verdict](reports/FINAL_VERDICT.md), [status](reports/STATUS.md), and [reproduction](REPRODUCIBILITY.md).';Path('README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote verdict, status, E2/E3 and complete handoff from actual evidence')


if __name__=='__main__':run()
