# 05 — Exact scoring, candidate formula, tests and reporting

## 1. Clean multi-choice scores without answer-generation confounds

For each model m and condition x, fixed task prompt q and four frozen options, compute teacher-forced log-likelihood of the literal *choice label* among A/B/C/D at the answer slot. Use the model's exact chat template and image processor with a full multimodal prompt; inspect token boundaries and verify that labels have valid, comparably tokenized continuations.

If labels are single tokens after the template, use the first answer-step logits directly. If not, use exact sequence log-probabilities for each complete label; avoid comparing unconstrained free-form answer strings with different lengths. Force matching separators/terminators; document the choice set normalization. Construct:
  
  p_m(a | x,q,A) = exp(s_m(a)) / Σ_{b in A} exp(s_m(b))

where s_m(a) is a properly conditioned continuation log score. This is probability *conditional on the four answer strings*, not calibrated probability over all possible natural-language answers. Verify max absolute numerical mismatch against a manual reference implementation on small synthetic examples.

For every image fix the ORIGINAL gold answer y and fixed candidate set across all edits. For any edited image, "gold" may no longer be semantically valid; our log score of the *original* y is a response-sensitivity measure, not the edited-image accuracy.

Use a numerically safe bounded logit contrast:

  L(x) = log p_m(y | x,q,A) - log(1 - p_m(y | x,q,A))

with probability clamped only for finite precision, e.g. [1e-6, 1-1e-6], and report the frequency of clamps. A sensitivity to removal operator o at region r:

  D_remove(o,r) = L(x_clean) - L(x_rem(o,r))

An analogous sham sensitivity:

  D_sham(o,r) = L(x_clean) - L(x_sham(o,r))

and sham-corrected difference:

  d_o(r) = D_remove(o,r) - D_sham(o,r)
         = L(x_sham(o,r)) - L(x_rem(o,r)).

Because shams do not necessarily perfectly match artifacts, d_o is NOT an identified unbiased semantic causal effect.

## 2. Proposed candidate estimator (v1, not asserted novel)

For an r candidate, a label-free off-target control r0 obtained without oracle support annotations, define:

  z_o(r) = d_o(r) - d_o(r0)

  M(r) = median_o z_o(r)
  V(r) = 1.4826 * median_o |z_o(r) - M(r)|

  S(r) = M(r) - λ * V(r)

where λ is chosen on development only from {0, 0.25, 0.5, 1.0}; primary candidate λ = 0.5 unless dev performance justifies a preregistered amendment. For multiple permissible off-target controls, average their d_o with predeclared fixed sampling seed; no cherry-picked least-sensitive edit.

A negative S has a concrete meaning: evidence of a stable removal-specific dependence is absent under the chosen contrast. Do NOT clip negative values when computing AUROC; clipping may artificially reduce rank variation.

Report separately M (median effect) and V (disagreement), because improving stability by subtracting dispersion can reduce true sensitivity.

As a key ablation replace median with mean, remove off-target term, remove sham term, replace V by standard deviation, or use only a single operator. This isolates what each component contributes. The equation and control construction are a *hypothesis*; if it fails, report failure and avoid post-hoc redefinition.

## 3. Oracle-region versus deployable region selection

In controlled diagnostic mode, evaluation annotations choose pairs (r+,r-) offline: scorer sees coordinates but cannot read which is the relevant region. This establishes whether the *score discriminates provided candidates*, not whether a system can discover critical regions in an arbitrary image.

For optional no-oracle mode, candidate regions come from a frozen shared proposal mechanism (e.g. precomputed SAM proposals or a fixed image grid), without reading GQA/RefCOCOg target labels. All baselines share proposals and run under the same edit budget. A sampled off-target "control" cannot be guaranteed irrelevant; audit that and disclose any labels used only afterward.

Do not build selection of candidates from the same VLM attention maps unless comparing them explicitly with properly matched baselines. It changes the inference problem.

## 4. Predeclared primary estimand and unit of analysis

Let each unique IMAGE i contribute a pair (r_i+, r_i-). For score S, define the paired correctness indicator I[S_i(r+) > S_i(r-)] + 0.5 I[equal]. Per-model/task score is its average.

Primary estimand = difference in this within-image pair-ranking accuracy between candidate S and the BEST BASELINE CHOSEN USING DEV ONLY, macro-averaged equally over four task×model cells (2 tasks × 2 models). This avoids inflated results due solely to easier task or larger model.

Uncertainty: paired cluster bootstrap over image IDs, with e.g. 2,000 replicate resamples stratified by task/model, seed 20264. Report paired 95% confidence intervals. Both models share images; for pooled CI resample image identities once within each task, carrying both model measurements together (do not assume model outputs are independent). Report per-cell CIs and component ablations. Use fixed one-sided hypothesis test only if specified before locking; otherwise use two-sided intervals.

If comparing multiple primary baselines after test inspection, primary claim invalid. Select one comparator with dev-only performance and keep all additional comparisons descriptive or control multiplicity with a preregistered Holm correction.

For pairwise AUROC, treat correlated per-image region scores appropriately and do not use naive independent region bootstrap.

## 5. Secondary reliability and failure diagnostics

- Operator agreement: Spearman correlations of rank/evidence contrasts on common region pairs; inversion probability across valid operator variants.
- Evidence artifact rate: irrelevant-region or sham intervention causing a large score relative to a predeclared dev threshold.
- Predictive quality: clean four-way accuracy, NLL, maximum confidence, ECE as descriptive with bins fixed in advance; Brier score.
- Selective decisions: expected error among retained original *unedited* predictions vs coverage; AURC/E-AURC, risk at 50%, 70%, 85%, 95% coverage. Include inference-call cost and assume oracle regions unavailable at deployment.
- Label leakage: image/text-only, shuffled image, choice permutation and image-only diagnostics.
- Edit validity: removal success, sham preservation, off-target validity and human reviewer agreement if available.
- Efficiency: model-forward counts, images/sec, NVML GPU memory, peak RAM and storage, wall time including image editing.

Do not interpret uncertainty intervals as magical correction for annotation noise, dataset leakage or adaptive test-set reuse.

## 6. Required failure and robustness reports

Report full dataset-eligible N, sample N, completed N, excluded N, invalid edit N, scored N, with reasons **for every operator, task and model**. Include both all-eligible (intention-to-evaluate with failure accounting) and validity-conditioned analyses, and report if filtering affects the model ranking.

Compute dev-only sensitivity to: perturbation severity, mask area, image resolution, prompt template, selection of incorrect baseline answers and repeated identical GPU inference. Do not change the confirmatory prompt, threshold or λ after seeing results.

Report that 120/task is a *pilot-sized confirmatory sample* and may not yield precise estimates for small effects; do not assert broad external validity or publication readiness from two model families alone.
