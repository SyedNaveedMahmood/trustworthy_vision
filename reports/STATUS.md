# Execution status

## Active amended development session — 2026-10-16 07:18–09:18 UTC

Branch safely fetched/pulled at `ef7c3e84211cfc3e424a5b9ec27adac109dfe31d`; original evidence below is archival. New records: `reports/evidence/v2/`, atomic checkpoints: ignored `outputs/v2/`.

- Qualification: Smol grounding rises 15/40 to 18/40 at 768 pixels but fails independent qualification (7/20, threshold 8/20). Official pinned InternVL2.5-2B runs in BF16 and scores 12/20 versus 7/20 text-only on separate RefCOCOg qualification. Qwen scores 11/20 versus 7/20.
- Dataset correction: 18/40 original GQA questions used a heterogeneous null-group distractor pool. Original GQA competence claims require withdrawal. Separate annotation-selected color diagnostic: Qwen 16/20 image, 6/20 text-only; InternVL 15/20, 7/20; Smol 15/20, 7/20. This supports only a narrow provisional Qwen+InternVL setup.
- Estimator algebra audited; mutual-negative MAD cancellation proved/tested. Amendment 08 defines sham, robust and candidate-specific independent-control alternatives before E2 scoring; oracle and frozen-prediction anchors remain distinct.
- 52 tests passed. Official LaMa CPU edits and Qwen/InternVL paired scoring finished for corrected pilot (15 GQA, 16 RefCOCOg; 775 score rows/model). E3 robust-independent versus raw mean: 0.0 percentage points, paired 95% CI [-5.0,+5.0]; versus sham-median: identical oracle pair rankings. No demonstrated added value from controls/dispersion. Annotation-free grid preparation and E4 image ablations are running. 360 blinded review packets prepared, zero human judgments: **HUMAN_AUDIT_PENDING**.
- E2–E4 are explicitly exploratory. E5 remains blocked; no confirmatory sampling or inference. Current verdict: **INCONCLUSIVE**.

Live counts, errors, checkpoints and resource usage: `reports/evidence/v2/execution_ledger.json` and per-run evidence. No process continuity is promised beyond the active session.

## Previous session (preserved)

2026-10-16 (Asia/Dhaka). Branch `pilot/controlled-evidence-v1`, based on main `853f02959495f2056fd3ffea56ecf9b7e8bf3d27`. No preexisting workspace or uncommitted user work.

- G0: source inventory recorded; both VLM revisions and weights, official LaMa source/checkpoint and dataset access resolved. Faithful closest-paper equation reproductions remain unfinished.
- E0/G1: computational smoke completed for both frozen BF16 models on 12 real images/task plus synthetic fixtures and occlusion variants. Finite likelihoods, exact repeatability and measured GPU fit. Full E0 PASS withheld because SmolVLM2 grounding debug competence is at chance and full editor semantic validity is unaudited.
- E1/G2: 40 separate development images/task, both models, clean/text-only/shuffled/rotated-choice conditions executed. Qwen: GQA 40/40, RefCOCOg 28/40. SmolVLM2: GQA 38/40, RefCOCOg 15/40. **Two-model gate failed:** 37.5% < 40% on SmolVLM2 RefCOCOg. E1 quantifies the flagged E0 weakness; it does not retroactively mark E0 passed.
- E2–E4: NOT EXECUTED. Competence gate failed; no eligible replacement was run. Operator human audit remains pending. Scoring/statistical core exists; full stage orchestration remains unfinished.
- G5/E5: BLOCKED, not locked. No confirmatory manifest sampled, no confirmatory scores, no comparator/lambda selected.
- E6: partial-run statistical reporting and failure audit completed. Scientific verdict: INCONCLUSIVE. Full-study E6 qualitative/primary-method audit remains unfinished with E2–E5.

Resolved failures: archived official RefCOCOg download verified; CUDA wheel download recovered with publisher checksum; Smol processor dependency repaired; official LaMa executed after protobuf/OpenCV repairs. No fake editor output. LaMa runs on isolated CPU, so the old torch does not use the RTX GPU.

Remaining blockers: prescribed model/task competence failure; official PaliGemma2-3B fallback config returns HTTP 401 and requires manual Hugging Face access; two independent human raters unavailable; ambiguity between label-free off-target controls and the 13-image oracle-negative reuse design. No inference that all possible alternative models are unavailable.

Evidence: `reports/evidence/`; local complete manifests, checkpoints, source snapshots and logs: ignored `outputs/`. No model process continues in the background after the session.
