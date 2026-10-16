"""Preserve raw/sham/control failures as genuine local original/edit panels."""
import json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
from src.storage import atomic_json


def run():
    failures=json.loads(Path('reports/evidence/v2/E4_failure_cases.json').read_text())
    manifest=json.loads(Path('outputs/v2/pilot_manifest_corrected.json').read_text())['items']
    edits=json.loads(Path('outputs/v2/edits_pairs_corrected.json').read_text())
    root=Path('outputs/v2/failure_panels');root.mkdir(parents=True,exist_ok=True);index=[]
    for failure in failures:
        task=failure['task'];iid=failure['image_id'];model=failure['model'];item=next(r for r in manifest if r['dataset']==task and r['image_id']==iid)
        panel=Image.new('RGB',(1600,1000),'white');d=ImageDraw.Draw(panel)
        for row,op in enumerate(['occlusion','blur','lama']):
            selected=next(e for e in edits if e['task']==task and e['image_id']==iid and e['region_index']==0 and e['operator']==op)
            control=next(e for e in edits if e['task']==task and e['image_id']==iid and e['region_index']==2 and e['operator']==op)
            for column,(label,path) in enumerate([('original',item['canonical_path']),('target removal',selected['removal_path']),('target sham',selected['sham_path']),('independent control removal',control['removal_path'])]):
                image=Image.open(path).convert('RGB');ImageDraw.Draw(image).rectangle(selected['box'] if column<3 else control['box'],outline='cyan',width=2)
                image=ImageOps.contain(image,(390,270));x=column*400;y=row*320+40;panel.paste(image,(x,y));d.text((x,y-20),op+' '+label,fill='black')
        d.text((4,4),f"{model} {task}/{iid} robust={failure['robust_pair_credit']} occlusion={failure['occlusion_pair_credit']} "+item['question'][:100],fill='black')
        path=root/f'{task}_{iid}_{model}.jpg';panel.save(path,quality=93);index.append({'model':model,'task':task,'image_id':iid,'path':str(path).replace(chr(92),'/'),'selection':'all_initial_pilot_robust_errors_or_raw_robust_rank_disagreements'})
    atomic_json('reports/evidence/v2/failure_panel_index.json',index)


if __name__=='__main__':run()
