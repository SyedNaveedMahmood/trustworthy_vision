# 02 — Datasets, tasks, reference evidence, partitions

## Two genuinely different visual tasks

### Task A: GQA question answering (natural-image reasoning)

Use **GQA balanced question data + official scene graphs** and source images. Official documentation:
https://cs.stanford.edu/people/dorarad/gqa/download.html

GQA exposes each image's scene graph, object bounding boxes, question/answer, answer types, word-to-object annotations and semantic programs. Use these as *candidate reference annotations*, not unquestionable causal ground truth. Audit every field and coordinate convention on a few examples before writing the loader.

Include: object identity/attribute questions, and a predeclared relation subset where the supporting evidence consists of the union of two objects. Restrict the primary region-pair analysis to questions with explicit, unambiguous referenced object IDs and an annotatable candidate region. For relation questions, define which object(s) must be removed and why. Exclude missing boxes, question leakage, tiny boxes, extreme crowding and unresolved contradictions with a public exclusion log.

Convert to a fixed multiple-choice format using 4 plausible answers **of the same answer type**. The reference answer is one choice; pick distractors without examining model predictions or confirmatory labels. Do not generate distractors conditioned on a model's desired errors. Freeze candidate strings and option order seed; optionally permute labels in a robustness test. Never use the original image/answer in a hidden chain-of-thought or answer-generating tool while preparing interventions.

Primary variable: probability assigned to the fixed reference answer among the four options for a given image and question. Also record model's chosen answer and correctness.

### Task B: RefCOCOg referring-expression comprehension (grounding)

Use the official RefCOCO/RefCOCOg annotation API at https://github.com/lichengunc/refer and associated MS-COCO images. Official TFDS instructions: https://github.com/tensorflow/datasets/blob/master/docs/catalog/ref_coco.md . RefCOCOg UMD splits partition images; Google's split may divide objects and hence can reuse images across partitions. Prefer UMD image-disjoint train/val/test and independently enforce unique COCO image IDs.

Present exactly four candidate boxes, one ground-truth expression referent and three matched distractors from the same image. Box letters A-D can be overlaid uniformly on *every version* of the image; text-only coordinate lists may instead be used if the model understands them. Freeze a single rendering protocol across all conditions. The original target box remains marked after removal (otherwise the task changes for an unrelated reason). Check box-overlap rules, object identity, target size, distractor plausibility, uniqueness and absence of easy label-position cues.

The task is selecting the referent among four boxes. It is not free-form bounding-box generation. Score only fixed A/B/C/D choices. A verified target-removal variant may make the question unanswerable; do not silently treat the original target option as a newly valid ground truth. Use its fixed-option probability change to measure dependence. A separate explicit NONE-OF-THE-ABOVE probe is exploratory and cannot be directly pooled with the four-choice metric.

### Dataset caveats

- GQA and RefCOCOg are not identical image distributions. Task-by-task results must be reported. Do not pretend "two tasks" means independent biological or driving domains.
- Object boxes are imperfect: a visible shadow/reflection or another same-category object can preserve evidence. Review candidate masks and verify target uniqueness.
- Source scene graph entities or referring expressions sometimes depend on more than one region. Keep multi-region cases separate instead of forcing single-object answers.
- Image licensing, access, downloading, checksum policy and reuse permissions must be recorded before fetching. Do not commit copyrighted images.
- If dataset download is impractical, document blocked status; do NOT silently substitute another benchmark and retain the same hypothesis.

## Fixed sample budgets

- Debug: 12 unique images/task, 2 tasks, visible to developers. Pure infrastructure checks, no scientific inference.
- Development pilot: 40 distinct unique images/task (80 total), balanced as feasible across predeclared types; used for tuning operator parameters and method hyperparameters.
- Confirmatory: 120 distinct unique images/task (240 total), locked and never used for method design or manual cherry-picking; **both frozen models see the same 240**. A minimum feasible target, not a powered guarantee of statistical significance.
- At most one primary analysis item per image in a partition. Keep original image IDs, question/ref IDs and image/content hashes.
- Keep partitions image-disjoint within each dataset. Check cross-dataset duplicate hashes and report if common COCO imagery creates overlap.
- Never select confirmatory samples based on whether one VLM answers correctly. Report analyses of all eligible items, then stratify by correctness.
- Task A strata targeted: object category, attribute, relation and spatial (30 per stratum when eligible); if single-region validity rules exclude a subtype, record the shortfall and analyze the actual sample instead of relabeling.
- Task B strata targeted: object people vs non-people, target size small/medium/large, relational vs appearance expressions. Use marginal strata rather than pretending these can all be crossed evenly with n=120.
- Maintain independent sampling seeds 20261 (debug), 20262 (dev), 20263 (confirm). Record dataset revision and sorted IDs and SHA-256 of the manifest.
- If a required stratum has insufficient clean annotations, report its eligible count, freeze available data and lower strength of claims. Do not duplicate images to reach 120.

## Manifests

Each row must contain: dataset name/version/split; image_id, image sha256, question_id or ref_id; original question/expression; four candidates; reference letter and canonical semantic answer; annotation object IDs; region polygons/boxes with coordinate system; task subtype; edit eligibility flags; sampling seed; partition; source license and URL; and human/automated audit status. Store private paths locally and relative dataset pointers only in public manifests.

Use schemas/ or unit tests to validate:
1. identical images in different partitions rejected;
2. four unique plausible candidates, exactly one correct for unedited case;
3. all region boxes clipped and transformed correctly after resize;
4. image content hash agrees with the image actually scored;
5. any reason for exclusion is recorded *before* seeing model output;
6. no gold relevant region can be read by inference-time proposal logic, except in explicit diagnostic-oracle mode.

If RefCOCOg annotations need a separate manual download, the agent must not fabricate a download URL.
