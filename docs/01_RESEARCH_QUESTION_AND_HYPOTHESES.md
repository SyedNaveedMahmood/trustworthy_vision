# 01 — Research decision, novelty boundaries, hypotheses

## Research question

Given a fixed frozen VLM and a fixed image-question-answer tuple, can we estimate whether the model depends on an independently annotated, task-relevant image region while separating sensitivity to semantic evidence removal from sensitivity to artifacts of the editing operation?

This is an *intervention sensitivity measurement* problem, not proof of causal dependence under unconstrained image manipulations. A counterfactual edit need not be on the natural-image manifold or preserve all other causes. Do not describe observationally calibrated scores as causal identification unless the assumptions and validations justify it.

### Scope of contribution to investigate

1. Quantify an editing-artifact failure mode of raw region-dropout evidence scores across operators and models.
2. Develop a robust region-evidence estimator based on target removal, region-preserving shams, independently sampled off-target edits, and operator consistency. This is a candidate method, not established novelty.
3. Test whether it ranks annotated supporting regions above matched non-supporting regions, survives held-out operator variants, and helps select reliable predictions beyond entropy at equal compute.
4. State precisely when the method fails: semantic inpainting errors, residual object cues, contextual relations, operator disagreement, language-prior answers, wrong baseline answers and multiple necessary regions.

Do NOT claim that region perturbation, inpainting, sham controls, multiple judges, robustness statistics, or selective answering individually are new.

## Closest prior before claiming novelty

- EVICT, Kotte, CVPR Workshops 2026: counterfactual evidence dropout for selective grounded QA; specifically reports tasks/settings where entropy wins. Official paper: https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Kotte_EVICT_Evidence-Sufficiency_Verification_via_Counterfactual_Dropout_for_Visually-Grounded_Selective_Question_CVPRW_2026_paper.html
- CoEV, MICCAI 2026: bidirectional image-text evidence verification with masking and correction. Official paper: https://papers.miccai.org/miccai-2026/0442-Paper0281.html . Authors' linked GitHub repository https://github.com/Zhounan1222/CoEV was checked on 2026-10-16 and had only a six-byte README, so it is NOT runnable code at the time of drafting.
- CSGR, Jojic et al., arXiv:2609.13228: counterfactual region search, realistic perturbations and multiple judges. https://arxiv.org/abs/2609.13228 . A complete code release was not independently confirmed.
- VLM Reality Check, CVPR Workshops 2026: many controlled image counterfactual challenges. https://openaccess.thecvf.com/content/CVPR2026W/DataMFM/html/Sar_VLM_Reality_Check_A_Causal_Counterfactual_Benchmark_for_Diagnosing_Cognitive_CVPRW_2026_paper.html
- A model score that wins only against raw black masks, not against strong realistic and sham-controlled prior methods, is NOT convincing evidence for a new method.

Reaudit paper versions, appendices, baselines, official code and contemporaneous 2026 work before finalizing claims. Mark each baseline as OFFICIAL_CODE, PAPER_REIMPLEMENTATION, PROXY or UNAVAILABLE. Never treat a proxy as a replication.

## Primary hypotheses (directional, testable)

H1, diagnostic: naive single-operator masking yields larger apparent evidence dependence on at least some annotated irrelevant regions and/or low-quality edits than a well-controlled intervention score. This must be demonstrated, not assumed.

H2, primary method: our robust artifact-controlled evidence index increases *within-image pairwise discrimination* between annotated relevant and area/texture-matched irrelevant regions relative to the strongest implementable reference score, averaging effects across tasks and models.

H3, secondary: the index exhibits better agreement (rank correlation and lower rank inversion) across valid editing operators, including a held-out edit variant, without trivially collapsing scores toward zero.

H4, exploratory: score-informed abstention yields lower wrong-answer selective risk at fixed coverage than entropy, max-choice-probability and single-operator dropout when intervention costs are budget-matched.

N1, negative control: editing an irrelevant region of comparable area/texture should not systematically reduce support for the correct answer as much as verified removal of its annotated relevant region. Failure is informative.

N2, sham control: an edit designed to preserve relevant content should not cause effects comparable to verified semantic removal. If it does, operator artifacts dominate and the apparent mechanism is unidentifiable.

H2 is the go/no-go primary result. H3 and H4 cannot rescue a failure of H2 by post-hoc relabeling.

## What the pilot can and cannot establish

Can: diagnose operator dependence, compare controlled scores, test robustness on two frozen model families, estimate selectivity and assess feasibility.

Cannot: establish "faithful visual explanations" for arbitrary free-form generations, unbiased causal effect of semantic evidence, medical reliability or a general region discovery algorithm from oracle ground-truth boxes. Ground-truth masks may be used to *evaluate the score* but must not leak into its inference-time region proposal or off-target sampler.

If this remains a score on given candidate regions, describe it as a controlled estimator / reliability test, not as automated grounding. Automated proposal ranking is a later ablation with label-free proposals shared by all baselines.

## Pre-registration

Freeze config, prompts, region sampler, operator implementations, score formula, metric definitions, success criterion, analysis scripts and hash of confirmatory image-ID manifests *before any confirmatory inference*. Dev and operator-validity audits may lead to a v2 amendment; record old and new values and do not erase failed versions.

Confirmatory negative results must be reported even if they are inconvenient.
