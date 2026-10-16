# E2 exploratory intervention results

**Real computational pilot executed; semantic validity remains HUMAN_AUDIT_PENDING.** The corrected cohort contains 15 annotation-selected harder color GQA and 16 original-development RefCOCOg images, shared by Qwen and InternVL. Each model has 775 complete score rows: clean plus 24 removal/sham/control images per item. Source-native images are capped at 768 without upscaling; RefCOCOg overlays are applied consistently after intervention. No confirmatory image was sampled or scored.

The original mixed-type GQA intervention preparation is preserved as a separate unscored legacy cohort. It is excluded from corrected E3 evidence because of the audited distractor defect. New cohort IDs, exclusions, box-area mismatches, independent-control overlaps and hashes are saved in `evidence/v2/pilot_sample_summary_corrected.json`; full local manifest is `outputs/v2/pilot_manifest_corrected.json`.

| Operator | Requests | Inside expanded-mask mean absolute RGB change | Outside-mask max change | Sham-core max change |
|---|---:|---:|---:|---:|
| Global mean fill | 124 | 53.23 | 0 | 0 |
| Gaussian blur, sigma 24 | 124 | 31.73 | 0 | 0 |
| Official LaMa | 124 | 34.28 | 0 | 0 |

Each request yields removal and sham: **372 requests, 744 edited images**. No identity removals or identity shams in this pilot. Mean core area is 7.83% and expanded-mask area 9.68%. Mask perimeter, exact output hashes, pixel differences and geometry are recorded per request. The 8px halo lets the sham retain an operator-induced boundary while restoring exact core pixels. This does not guarantee matched semantic artifacts: contextual evidence can change in the halo; blur can retain color or shape; inpainting can reconstruct or hallucinate objects. Off-target controls have equal candidate dimensions and are selected from image geometry/seed alone. They can overlap relevant contextual content; their validity is unknown.

LaMa ran the unmodified official predictor from pinned source `786f5936b27fb3dacd2b1ad799e4de968ea697e7` and the pinned downloaded big-lama checkpoint in its isolated Python3.8 CPU environment. Completed raw predictions are reused by content/mask/operator hash. The corrected batch needed 61 remaining requests after reuse of previously completed RefCOCOg edits; the original legacy preparation and CPU wall times remain recorded separately. No substitute inpainting was used. Outputs are composited back to original pixels outside the mask; raw LaMa outside-region change was also zero in this batch.

Measured paired-score model process resources (includes load, preprocessing, cache and atomic checkpoint I/O): Qwen **151.75s**, peak NVML **8.83 GiB**, Torch allocated **7.57 GiB**; InternVL **101.62s**, peak NVML **7.10 GiB**, Torch allocated **5.23 GiB**. NVML includes desktop usage and is sampled at 100ms; short spikes can be missed. Detailed reserved/RSS peaks, input tensors, image-token counts and per-forward timings are in the score files. BF16, batch one, sequential model process lock; LaMa never occupies GPU memory.

## Review material and examples

**360 randomized packets** are ready: 30 per operator × candidate/control role × removal/sham condition. `outputs/v2/human_review/review_form.csv` contains empty reviewer fields for removal validity, sham preservation, off-target validity, unexpected semantics, artifacts and confidence. Independent raters must complete it. The separate private key maps blinded IDs to conditions; keep it hidden while reviewing. **Zero ratings and zero independent reviewers have been recorded.**

Genuine local examples: `outputs/v2/human_review/contact_sheet_01.jpg` (12 randomized original/edit pairs), all individual `V2_*.jpg`, and further contact sheets. Dataset-derived image files are preserved locally rather than uploaded to GitHub. Agent visual inspection of the first sheet shows visible filled/blurred patches and shams retaining recognizable content, but is not an independent human audit or a validity pass. Reviewers need the intended-region pointer and both original/edited full-resolution images for ambiguous cases; the packet is being strengthened accordingly.

E3 can summarize provisional numerical sensitivities, and E4 can diagnose artifacts, while E2 semantic validity remains unresolved. A four-grid-candidate, predicted-answer diagnostic is in progress and is distinct from these oracle-region pairs. E2 is **PARTIAL / EXPLORATORY**, not officially passed.
