# 03 — Image interventions, controls, operator validity

## Intervention variables and notation

For each clean image x, question q, fixed candidate answers A and proposed region r:
- REM(o,r): image after attempting to remove the semantic evidence in region r with operator o.
- SHAM(o,r): operator-specific control altering image around r while attempting to preserve the relevant semantic content.
- OFF(o,r-,area): matched perturbation to another candidate region r- selected by a predeclared label-free region sampler.
- IDENTITY: byte-identical image through the entire preprocessing/scoring pipeline.

Ground-truth r+ is held in evaluation metadata and cannot be used by a deployed region proposal method. In the **oracle-region diagnostic only**, choose r+ and matched r- from the frozen human/annotation manifest to measure whether a proposed evidence score discriminates them. Label diagnostic-oracle results prominently; they do not show automatic localization.

## Three base removal operators (frozen in development)

O1, rectangular or mask-shaped occlusion: fill with local color mean or fixed neutral gray; choose ONE as the preregistered default, retain the other as sensitivity. Very cheap but creates obvious boundaries and distribution shift.

O2, strong Gaussian blur inside target with a feathered boundary. Blur kernel/sigma chosen in dev; classify as valid removal ONLY when an independent check finds the diagnostic object/attribute unreadable. Blur may fail to erase coarse evidence.

O3, LaMa semantic inpainting: use the official WACV 2022 code at https://github.com/advimman/lama pinned to a source commit and checkpoint hash. Give it the same mask and resize protocol. Inpainting can hallucinate a replacement object or distort semantic context; log those cases. This code has older dependencies; isolate in its own environment/process if necessary. A failed dependency install does not justify replacing the output with a fabricated result.

Do not add a diffusion-based editor to the primary three-operator panel. It is costlier, raises prompt leakage and may insert semantic content. A fourth diffusion operator may be a holdout experiment only after reproducibility and funding checks, with a protocol amendment.

## Operator-specific shams: imperfect, falsifiable controls

Black/mean occlusion: draw a contour/border artifact matched to the mask boundary while leaving the target contents present. It does not perfectly match the interior removal artifact; disclose this.

Gaussian blur: use a narrow halo/boundary blur or a mild content-preserving version whose semantic-preservation rate is independently checked. Do not claim the artifact magnitude is exactly matched to strong blur.

LaMa: run inpainting and then alpha-composite the original foreground or annotated region back into the inpainted image using the same feathering rule; also use a separate identity/no-op round trip. This may introduce seams. Verify preservation rather than assuming it.

For every operator include a spatially matched off-target edit, sampled **without access to which region is correct** in the inference code. For oracle diagnostics, construct an independent annotated irrelevant candidate only as an evaluation reference. Match area within ±15% where possible, approximate texture/edge density and avoid overlap. If no suitable off-target candidate exists, record ineligible; never pick the least sensitive region after observing model outputs.

Apply identical resizing, option box overlays, color handling, JPEG quality and display crops to clean/removal/sham/off-target images. Score the same reference answer and choice order after every edit.

## Edit artifact and information removal auditing

Before confirmatory runs, create a blinded grid/contact-sheet and an independent edit validity log:
- confirm the correct region was modified;
- assess whether task-relevant object/attribute/relation is absent on removal;
- assess whether it remains on sham;
- confirm off-target editing did not remove another necessary reasoning object;
- assess whether editing introduced a new object, text, false cue, crop or severe border defect;
- record mask percent of image, mask perimeter/area, edge contrast, image-resolution changes and SSIM/LPIPS as *diagnostics*, not substitutes for semantic validity.

At least 30 randomly chosen edits/operator/condition in dev should receive blind review by two independent humans if available. Separate disagreements and adjudication. If human review cannot be performed, mark human gate PENDING and call the computational analysis provisional; do not silently mark it passed.

Operator eligibility thresholds (design targets, not observed facts): >=90% removal success and >=95% sham semantic preservation for audited images; >=95% off-target noncritical validity. If an operator fails, change only on dev with amendment, or exclude it *globally* from confirmatory analysis with a documented downgrade. Never drop individual confirmatory examples because their model output is inconvenient; invalid edits remain in the intention-to-evaluate accounting and receive sensitivity analysis.

## Image-edit factorial per sample

For each r+ (oracle positive) and r- (reference negative):
- Clean: one canonical image, verify repeated forward is numerically stable.
- O1/O2/O3: REMOVE and SHAM on each region (3 x 2 x 2 = 12 variants).
- Globally matched context correction: off-target controls are already included by r-; optionally run an additional unrelated background mask for sensitivity.
- Holdout operator variant: do NOT use its labels/scores in development; evaluate operator transfer by predicting outcomes or ranks from the training operators.
- Small dose-response on a predefined 20-item dev subset: mask dilation factors 0.75, 1.0, 1.25; measure semantic validity and score drift.

Budget: 13 images/sample for base factorial (12 edited + clean), plus held-out/dose subset. 240 confirmatory samples × 2 models × 13 images = **6,240 model-image scoring events** before additional baselines, choice permutations and retries. This is a count, not a runtime prediction. Cache per-(model, prompt, image hash, options, processor config) logits and reuse across all baseline scores.

## Failure cases that must be preserved

Occlusion object still inferable via shadow, text or a duplicate instance; overly large box erases another needed entity; "relevant" target annotated incorrectly; inpaint substitutes a plausible replacement; LaMa failed; option overlay shifted; foreground compositing introduces a strong seam; sham accidentally deletes information; first VLM answer incorrect; "off target" is actually contextual evidence.

Log these as distinct categories. They may be the central finding rather than noise to delete.
