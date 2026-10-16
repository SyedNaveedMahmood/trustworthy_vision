"""Pinned official InternVL2.5 model; preprocessing adapted from MIT model-card example.

Copyright (c) 2024 OpenGVLab. Official source retained in the HF snapshot, not vendored.
Dynamic tiles are capped at 4 plus thumbnail as an explicit 16GB compute factor.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel,AutoTokenizer
from filelock import FileLock
from .choice import ChoiceModel,Budget
from ..storage import digest


def official_tiles(image,max_num=4,image_size=448):
    w,h=image.size; aspect=w/h
    ratios=sorted({(i,j) for n in range(1,max_num+1) for i in range(1,n+1) for j in range(1,n+1) if i*j<=max_num},key=lambda x:x[0]*x[1])
    best=(1,1); diff=float('inf')
    for ratio in ratios:
        d=abs(aspect-ratio[0]/ratio[1])
        if d<diff or (d==diff and w*h>.5*image_size**2*ratio[0]*ratio[1]): best,diff=ratio,d
    resized=image.resize((image_size*best[0],image_size*best[1]))
    tiles=[resized.crop((i%best[0]*image_size,i//best[0]*image_size,(i%best[0]+1)*image_size,(i//best[0]+1)*image_size)) for i in range(best[0]*best[1])]
    if len(tiles)!=1: tiles.append(image.resize((image_size,image_size)))
    return tiles


class InternChoiceModel(ChoiceModel):
    def __init__(self,long_edge=768):
        meta=json.loads(Path('reports/evidence/v2/internvl_source_probe.json').read_text())
        self.model_id=meta['model_id']; self.revision=meta['revision']; self.family='internvl'; self.long_edge=long_edge
        self.process_lock=FileLock('outputs/primary_model.lock',timeout=0); self.process_lock.acquire()
        self.model=AutoModel.from_pretrained(self.model_id,revision=self.revision,code_revision=self.revision,
                  trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False).eval().cuda()
        self.model.requires_grad_(False)
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_id,revision=self.revision,code_revision=self.revision,trust_remote_code=True,use_fast=False)
        self.model.img_context_token_id=self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.transform=T.Compose([T.Resize((448,448),interpolation=InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
        self.processor_config={'long_edge':long_edge,'dtype':'bfloat16','batch_size':1,'max_dynamic_tiles':4,'thumbnail':True,
                               'tile_size':448,'attention':'eager_official_fallback','adapter_sha256':digest(Path(__file__).read_bytes()),'official_code_revision':self.revision}
        self.budget=Budget()

    def set_long_edge(self,edge): self.long_edge=edge; self.processor_config['long_edge']=edge

    def encode(self,image,question,options):
        if len(set(options))!=4: raise ValueError('Four distinct labels/options required')
        prompt=question+'\n'+'\n'.join(f'{a}. {b}' for a,b in zip('ABCD',options))+'\nAnswer with only the letter A, B, C, or D.'
        pixels=None
        if image is not None:
            image=image.convert('RGB').copy(); image.thumbnail((self.long_edge,self.long_edge))
            tiles=official_tiles(image)
            pixels=torch.stack([self.transform(im) for im in tiles]).to(device='cuda',dtype=torch.bfloat16)
            prompt='<image>\n'+prompt
        conv=self.model.conv_template.copy(); conv.system_message=self.model.system_message
        conv.append_message(conv.roles[0],prompt); conv.append_message(conv.roles[1],None)
        rendered=conv.get_prompt()
        if pixels is not None: rendered=rendered.replace('<image>','<img>'+'<IMG_CONTEXT>'*(self.model.num_image_token*len(pixels))+'</img>',1)
        encoded=self.tokenizer(rendered,return_tensors='pt')
        inputs={k:v.cuda() for k,v in encoded.items() if k in ['input_ids','attention_mask']}
        inputs['pixel_values']=pixels
        if pixels is not None: inputs['image_flags']=torch.ones((len(pixels),1),dtype=torch.long,device='cuda')
        prefix=self.tokenizer.encode(rendered,add_special_tokens=False); labels=[]
        for label in 'ABCD':
            seq=self.tokenizer.encode(rendered+label,add_special_tokens=False)
            if seq[:len(prefix)]!=prefix or len(seq)!=len(prefix)+1: raise RuntimeError('InternVL label continuation is not one token')
            labels.append(seq[-1])
        self.last_preprocess={'native_input_size':list(image.size) if image is not None else None,
                              'tensor_shapes':{k:list(v.shape) for k,v in inputs.items() if v is not None},
                              'image_token_count':int((inputs['input_ids']==self.model.img_context_token_id).sum().item())}
        return rendered,inputs,labels

    def forward_inputs(self,inputs):
        if inputs['pixel_values'] is None:
            return self.budget.forward(self.model.language_model,**{k:v for k,v in inputs.items() if k!='pixel_values'})
        return self.budget.forward(self.model,**inputs)
