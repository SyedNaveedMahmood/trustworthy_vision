"""Conservative VRAM admission/headroom guards for the measured batch-one regime."""
GIB=1024**3
MODEL_FREE_REQUIREMENTS={'qwen':10*GIB,'smol':7*GIB,'internvl':8*GIB}


def check_memory(free,total,required_free,phase):
    if free<required_free or total<required_free:
        raise MemoryError(f'VRAM headroom policy blocked {phase}: free={free}, total={total}, required_free={required_free}')
    return True


def admit_model(family):
    import torch
    free,total=torch.cuda.mem_get_info()
    check_memory(free,total,MODEL_FREE_REQUIREMENTS[family],'model loading')


def guard_forward(phase):
    import torch
    free,total=torch.cuda.mem_get_info()
    check_memory(free,total,(2 if phase=='before' else 1)*GIB,phase+' forward')
