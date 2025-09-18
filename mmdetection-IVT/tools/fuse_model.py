import torch

pretrained_vi = torch.load('/raid/liufangcen/updata/checkpoint/convmae_base.pth')['model']
pretrained_ir = torch.load('/raid/liufangcen/updata/checkpoint/checkpoint-399.pth')['model']

dict = {}
for k, v in pretrained_ir.items():
    if 'blocks3' in k:
        key = k.replace(k.split('.')[2], k.split('.')[2] + '_y')
    else:
        key = k.replace(k.split('.')[0], k.split('.')[0] + '_y')
    dict[key] = v
pretrained_vi.update(dict)
torch.save(pretrained_vi, '/raid/liufangcen/updata/checkpoint/merged_convmae_infmae.pth' )