"""Hash and factorial integrity checks for resumed edited-image experiments."""
from pathlib import Path
from ..storage import digest


def validate_scoring_inputs(items,edits,mode='pairs'):
    lookup={(i['dataset'],i['image_id']):i for i in items};cells=set();selected=set()
    for e in edits:
        identity=(e['task'],e['image_id']);cell=identity+(e['region_index'],e['operator'])
        if identity not in lookup or cell in cells:raise ValueError('Unknown or duplicate edit cell')
        cells.add(cell);selected.add(identity)
        for c in ['removal','sham']:
            if digest(Path(e[c+'_path']).read_bytes())!=e[c+'_sha256']:raise ValueError('Edited-image hash mismatch')
        if e['outside_mask_max_change']!=0 or e['sham_core_max_change']!=0:raise ValueError('Pixel locality invariant failed')
    nregions=4 if mode=='pairs' else 8
    expected={(t,i,r,o) for t,i in selected for r in range(nregions) for o in ['occlusion','blur','lama']}
    if cells!=expected:raise ValueError('Incomplete edit factorial')
    for identity in selected:
        item=lookup[identity]
        if digest(Path(item['canonical_path']).read_bytes())!=item['canonical_sha256']:raise ValueError('Canonical-image hash mismatch')
    return {'selected_images':len(selected),'edit_requests':len(cells),'expected_score_rows':len(selected)*(1+nregions*3*2),'edit_manifest_sha256':digest(edits)}
