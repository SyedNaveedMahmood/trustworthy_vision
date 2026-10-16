# 04 — Experimental matrix and operational acceptance gates

All studies use two frozen, distinct-family VLMs and the same versioned input manifest. Never train a detector or tune language-model weights in the pilot. Every study records actual forward count, cache hit rate and wall/GPU peak. Confirmatory tasks/variants cannot be inspected or hand-tuned before the protocol lock.

## E0 — Infrastructure and source reproduction (no scientific conclusion)

Question: can established model/dataset/editor code load and produce correctly aligned results on an RTX 4080 SUPER 16 GB?

Inputs: 12 debug examples per task; at least four unedited and four edited images of each task per model.

Verify: CUDA/VRAM, GPU architecture; model weights + tokenizer/processor revisions; deterministic scoring and candidate-label permutations; annotation coordinates after resize; inpainting replay hash; dataset manifests; ability to reconstruct task label exactly; output cache and atomic checkpoints.

Compare teacher-forced choice scores with independent manually reviewed next-token candidate examples. Synthetic image tests: red/blue shapes, deliberately unique targets, border masks and shuffled letters. **Pass** only if clean logits finite, options map correctly, masks exactly located, schema valid, no result contamination, and each model gets > chance-level signal on 4-way debug (check later on dev as an accuracy gate). Record original-source licenses and commits. No pipeline may silently substitute fake/generated data on network failures.

Deliver: manifests/debug, tests, environment/versions, an E0 report with PASS/BLOCKED/FAIL and evidence.

## E1 — Clean-task competence and scoring sanity (development)

Question: do the frozen models have enough clean visual skill to make intervention scores interpretable?

40 dev images per task seen by both models; four frozen options each. Compare with text-only/no-image and shuffled-image prompts; include option-letter permutation checks.

Metrics: 4-way clean accuracy, NLL of correct option, probability mass calibration (descriptive), all-option entropy, image-versus-no-image gain, label permutation sensitivity, percent invalid completions. Subgroup breakdown by question type and target area.

Predeclared *feasibility gate*: >=40% clean accuracy on each of the two four-way tasks (chance = 25%) per model, plus evidence of image-over-text improvement. This is an engineering screen, not a significance test or paper result. If SmolVLM2 fails, test an independently trained 2B-4B fallback that fits measured 16 GB with the same frozen protocol and report substitution as a preregistered model change; otherwise mark 2-backbone claim blocked rather than selecting the best-performing examples.

## E2 — Intervention validity and artifact diagnosis (development)

Question: do different removal operators and shams actually behave as intended?

Use all 40 development images per task with full operator factorial. Produce blinded contact sheets and structured audit; at least 30 audited edits/operator/condition if feasible. Examine score shifts for irrelevant-region removal, sham edits and identity round-trips. Report intended-removal failure rates before analyzing model success.

Metrics: semantic removal/preservation audit, mask area, edge distortion, no-op disagreement, off-target positive shifts, Pearson/Spearman rank association of raw sensitivity with mask area, relation subset failure rates, operator agreement.

Pass to E3 only with validated operators and an explicit list of failed controls. No valid sham means **mechanistic claim is blocked**, although the artifact-diagnosis analysis may continue as descriptive work.

## E3 — Baselines and proposed estimator (development, tuning permitted)

Question: are artifact and irrelevant-region effects measurable, and can an explicitly controlled estimator reduce them?

Baselines, in fixed order:
B0: no-image language-prior score (diagnostic, not a localization score).
B1: unedited choice entropy / max confidence (prediction reliability).
B2: raw blackout/mean occlusion difference, EVICT-like single operator.
B3: individual blur and LaMa evidence removal scores.
B4: simple multi-operator mean/median without controls.
B5: best feasible sham-corrected control-only estimator.
B6: best feasible off-target-corrected estimator.
B7: strongest faithful implementable literature baseline, with OFFICIAL_CODE versus REIMPLEMENTATION label. Add CSGR/CoEV/BCEA only when task and implementation match; their scores are not interchangeable with region ranking by assumption.
Candidate: artifact-corrected operator-aggregated evidence index specified in 05, with coefficient and robust aggregation selected on dev only.

Primary diagnostic on candidate pairs: score annotated r+ and matched r- and predict which is the relevant region **using exactly the same intervention access** for every compared method. Metrics: paired choice accuracy, pairwise AUROC, false-positive rate on negative regions, operator dispersion, bootstrap CIs. Report easy versus hard pairs by area and image type.

Gate: the proposed index must outperform the strongest B2-B6 comparator by a practically meaningful dev margin (design heuristic ≥0.03 pairwise accuracy/AUROC) *without* extra non-accounted editing cost, or there is no reason to proceed to a method claim. If not, finish an honest negative pilot rather than repeatedly tuning on test.

## E4 — Operator transfer and budget ablations (development / frozen tests)

Question: is the effect an invariant rather than an overfit to particular editing artifacts?

Freeze final score after development, then evaluate (a) held-out removal variant: e.g. alternative fill color or blur strength not used during calibration; (b) leave-one-operator-out aggregation; (c) sham removal; (d) off-target removal; (e) mean vs median vs robust median-minus-dispersion; (f) choice-letter permutations; (g) mask size stress; (h) artifact-quality strata; (i) substitute region boxes with automatic proposals as explicitly NON-oracle study if computationally feasible.

Metrics: rank Spearman correlations and pairwise inversions across operators; retained primary AUROC; variance; failure types; forward calls and latency.

Falsifier: robust score merely becomes smaller or more constant for all regions and thereby *appears* stable but loses discrimination. Stability without discriminative power is not success.

## E5 — LOCKED confirmatory generalization

Freeze specs, hash manifests, forbid post-hoc threshold choice. Run 120 previously unseen images/task, both backbones, same operator factorial. Calculate preregistered primary effect: task/model macro difference in within-image relevant-versus-negative pair ranking accuracy between proposed method and strongest dev-selected baseline, using image-cluster bootstrap confidence interval and paired randomization or sign test. Also compute task/model-level paired AUROC/PR-AUC where well-defined; report intervals not only p-values.

Secondary: operator-transfer agreement and false-evidence rate. Exploratory: image-level selective risk at 50/70/85/95% coverage using only labels on original unedited images (and separately on audited edited images), compare entropy, confidence and evidence scores. Thresholds chosen on dev. Compute E-AURC/AURC, AUROC for erroneous predictions, and correctness/grounding diagnostics separately. Report selective risk only for a clearly defined no-oracle deployable score; oracle-region scores may be shown as a labeled upper-bound diagnostic, NOT a deployment result.

SUCCESS/NO-GO: require positive pooled paired effect with uncertainty interval excluding zero and no catastrophic reversal on one entire task/model, plus actual audit validity. If pooled effect does not clearly exceed zero, mark INCONCLUSIVE or NEGATIVE. n=120/task may be underpowered; a preliminary positive finding warrants later independent replication, not an inflated paper claim.

## E6 — Reproducibility, error analysis and independent audit

Generate: full per-item records; versioned configs and dataset/image/checkpoint hashes; baseline status matrix; runtime/VRAM summaries; operator failure plots; region-pair ROC and bootstrap intervals; 20 predeclared qualitative panels (random success and failure, not cherry-picked); protocol deviations and failed runs; robust controls; evidence for all exclusion decisions; exact commands.

Review failure buckets: residual cues, contextual dependencies, irrelevant-region effects, language-prior answers, image-preprocessor changes, input-label imbalance, off-target leakage and synthetic edit hallucinations. Run one blind reviewer pass of qualitative panels if humans are available. If not, document unavailable audit.

End report must answer: are the two VLMs usable; is artifact confounding real; do controls fix it; does the method outperform strong existing variants; does performance generalize; what remains unverified; should the paper proceed?
