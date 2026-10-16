"""Generate paired qualification diagnostics from preserved real forwards."""
import json,collections
from pathlib import Path
import numpy as np
from src.evaluation.v2 import paired_difference
from src.storage import atomic_json


def run():
    result={'development':{},'qualification':{},'color_diagnostic':{}}
    table=['| Model | Development condition | Correct / 40 | Prediction A/B/C/D | Mean image tokens | Mean score seconds |','|---|---|---:|---|---:|---:|']
    for model in ['qwen','smol']:
        record=json.loads(Path(f'reports/evidence/v2/qualification_{model}.json').read_text())
        cells={}
        for condition in sorted({r['condition'] for r in record['rows']}):
            rows=sorted([r for r in record['rows'] if r['condition']==condition],key=lambda r:r['image_id'])
            counts=collections.Counter(r['prediction'] for r in rows)
            cells[condition]={'correct':sum(r['correct'] for r in rows),'n':len(rows),'predictions':dict(counts),
                'gold_letters':dict(collections.Counter(r['gold'] for r in rows)),
                'mean_score_seconds':float(np.mean([r['wall_seconds'] for r in rows])),
                'image_token_counts':sorted({r['preprocessing']['image_token_count'] for r in rows}),
                'mean_image_tokens':float(np.mean([r['preprocessing']['image_token_count'] for r in rows]))}
            table.append(f"| {model} | {condition} | {cells[condition]['correct']} | "+'/'.join(str(counts[l]) for l in 'ABCD')+f" | {cells[condition]['mean_image_tokens']:.1f} | {cells[condition]['mean_score_seconds']:.4f} |")
        original={r['image_id']:r for r in record['rows'] if r['condition']=='legacy_512_image'}
        comparisons={}
        for condition in cells:
            rows=sorted([r for r in record['rows'] if r['condition']==condition],key=lambda r:r['image_id'])
            comparisons[condition]={**paired_difference([int(r['correct']) for r in rows],[int(original[r['image_id']]['correct']) for r in rows]),
                                    'changed_predictions':sum(r['prediction']!=original[r['image_id']]['prediction'] for r in rows)}
        result['development'][model]={'cells':cells,'paired_vs_512_legacy':comparisons,'last_process_telemetry':record['telemetry'],
                                     'telemetry_limitation':'Resume telemetry can cover only the final resumed process. Per-row wall times belong to original forward/cache computation; no condition-specific peak attribution.'}
    for model in ['qwen','smol','internvl']:
        hold=json.loads(Path(f'reports/evidence/v2/holdout_{model}.json').read_text());result['qualification'][model]={'metrics':hold['metrics'],'telemetry':hold['telemetry']}
        hard=json.loads(Path(f'reports/evidence/v2/hard_gqa_{model}.json').read_text());result['color_diagnostic'][model]={'metrics':hard['metrics'],'telemetry':hard['telemetry']}
    atomic_json('reports/evidence/v2/model_qualification_summary.json',result)
    text='''# Model qualification V2

**Provisional choice: Qwen2.5-VL-3B + InternVL2.5-2B.** The unchanged engineering screen is 40% image accuracy and positive image-versus-text gain. This is a small-sample feasibility screen, not a statistical proof of visual reasoning. SmolVLM2 remains unqualified on independent grounding; it was not discarded solely because of its original one-example deficit.

## Grounding development diagnostics

All conditions share the original 40 development images. Each comparison changes resolution or presentation alone relative to its stated reference. Legacy 512→768 scores: Qwen 28→30; Smol 15→18. 1024 gives the same accuracies: native COCO sources are at most 640 pixels, and the wrapper does not upscale them. The official processor then performs its own resizing/patching. Consequently, requested cap is not actual image detail or token count.

'''+ '\n'.join(table)+'''

Outer-letter rotation changes text answer positions and remaps the gold answer while keeping visual letters fixed. Visual rotation changes labels in the image and keeps the A–D text ordering. Numbered regions use inner 1–4 and outer A–D. These distinguish binding from position effects; none provides a clear, independently validated fix for Smol. B frequency alone cannot establish a token-scoring defect: exact label continuations are checked and manual probability and repeat checks passed. Correct/incorrect examples are preserved locally at `outputs/v2/diagnostic_panels/`.

## Independent qualification

20 RefCOCOg images are disjoint from original debug/development and were never used for selecting presentation. At legacy 768: Qwen **11/20 vs 7/20 text-only**; Smol **7/20 vs 5/20**; InternVL **12/20 vs 7/20**. Smol fails the fixed 8/20 minimum. These n=20 estimates are imprecise; refer to saved item-level outputs rather than extrapolating population competence.

The original GQA qualification reused an invalid mixed-type distractor recipe and is retained only as a diagnostic. The separate annotation-selected, image-disjoint basic-color set gives Qwen **16/20 image, 6/20 text, 4/20 shuffled**; InternVL **15/20, 7/20, 5/20**; Smol **15/20, 7/20, 8/20**. This qualifies a narrow color-identification development task; broad GQA competence remains unestablished. Its selection did not inspect model outputs, but it is development evidence, not confirmation.

## Official fallback and resources

[Official InternVL checkpoint](https://huggingface.co/OpenGVLab/InternVL2_5-2B), pinned revision `573169ee54df216786bb9a189e9a32a060a008cf`; model card declares MIT and embedded InternLM code carries Apache-2.0 headers. Source and weight hashes are recorded in `internvl_source_probe.json` and `internvl_download.json`. Official dynamic tiling is adapted with a fixed maximum four tiles plus thumbnail, 448 pixels; no quantization, BF16, eager attention because FlashAttention is absent. Qwen/Smol retain their official processors and SDPA. Models run sequentially under a process lock. The 4-tile cap is an explicit compute adaptation, not an official benchmark reproduction. No gated PaliGemma retry occurred.

Full-process peaks and times are in the machine-readable summary; image-token counts and original per-score wall times are in the table. Development Smol resume telemetry covers only the final resumed process and is not a reliable full-grid peak. NVML includes Windows desktop memory; Torch allocated/reserved peaks describe the process allocator. InternVL qualification measured 8.04 GiB device usage, leaving substantial headroom on the 16 GiB card.

**Gate assessment:** two distinct backbones meet the provisional corrected-task engineering screen. This does not clear semantic intervention validation, independent human audit, amended confirmatory lock, or full E0/E1 scientific gates. E5 remains blocked.
'''
    Path('reports/MODEL_QUALIFICATION_V2.md').write_text(text,encoding='utf-8')


if __name__=='__main__':run()
