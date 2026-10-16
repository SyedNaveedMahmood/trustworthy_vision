# Dataset and prompt audit

**The original GQA choice recipe contains a material validity defect.** 18/40 development questions have null global answer groups; pooling all null answers mixes binary answers, actions, locations and objects. An example is “Are there any boys on the snowboard?” paired with “no”, “taking notes”, “taking a photo”, and “below”. Syntax alone can eliminate distractors. The original 40/40 and 38/40 image scores do not establish broad visually grounded competence. No prior record was deleted or retrospectively repaired.

The saved audit includes global group counts, original correct-letter counts, binary questions (12/40), and literal answer occurrences (3/40). Existing image/text-only comparisons demonstrate language shortcuts but cannot determine necessity for each individual image: a text-only error does not prove that image content is causally necessary. Singular/plural and synonym answer alternatives also require semantic review.

## Separate harder diagnostic

`prepare_hard_gqa_v2` uses seed 20268 and annotation-only eligibility: global color, structural query, fixed eleven basic-color vocabulary, exactly one program object, answer confirmed in its scene-graph attributes, no literal answer in question, minimum 24px target and 1–35% target area. Distractors are different basic colors, preferring colors annotated elsewhere in the image. Every decision/exclusion is reproducible; no model accuracy is consulted. This creates 20 new development images from 2,306 eligible unique images. It narrows task scope to color identification and does not solve all language priors or annotation errors.

Real image/text-only/shuffled accuracy: Qwen 16/20, 6/20, 4/20; InternVL 15/20, 7/20, 5/20; Smol 15/20, 7/20, 8/20. These support visual dependence at the cohort level, with small-sample uncertainty. The corrected E2 sample contains 15 eligible images from this cohort, selected by fixed geometry rules before intervention outcomes. Qualification of the narrowed diagnostic and E2 exploration share this development cohort; neither is confirmatory evidence.

## RefCOCOg checks

Across 40 original development images: 0 image/annotation mapping errors; 0 identical first-sentence expressions mapped to multiple targets; 1 targets overlap a distractor at IoU >0.5. Exact duplicate-expression checks cannot establish semantic target uniqueness. Target area ranges 0.0519–0.8075 of image. Overlays alter a mean 0.092 and maximum 0.198 of target pixels, a real potential visibility confound. Bounding-box geometry and source mappings are valid; expressions, contextual negatives and label legibility remain subject to human review.

The v1 512px cap discards detail before official processing. V2 compares 768 and 1024 without artificially upscaling source images. Numbered regions, answer-position rotation and visual relabeling are separate conditions. All variants maintain the same coordinate transformation. Correct/incorrect prediction panels are local: `outputs/v2/diagnostic_panels/`; blinded edit panels: `outputs/v2/human_review/`. Dataset-derived images are not uploaded to GitHub.

## Split integrity

164 images across original debug/development, new qualification and harder development were checked by dataset/image identity, byte SHA-256, known Visual Genome→COCO mappings, decoded pixel hashes, and 64-bit DCT perceptual hashes. Observed hard duplicate flags: 0; perceptual distances <=4: 0. 47 GQA images lack COCO mappings. Null identities and pHash cannot prove photographic independence; these are bounded computational checks. No confirmatory manifest was generated or model-scored. GQA questions/scene graphs used for annotation eligibility are not confirmatory outcome access.

The 31-item E2 pilot has annotation-based target/negative pairs, not externally proven causal relevance labels. Negatives are nonoverlapping and within 50% target area; independent controls use geometry/seed only and may overlap other meaningful content. Report area mismatch and known overlaps rather than removing failures after seeing model effects. **HUMAN_AUDIT_PENDING** for semantic judgments.
