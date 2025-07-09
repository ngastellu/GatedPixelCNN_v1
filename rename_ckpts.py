#!/usr/bin/env python

import torch
from pathlib import Path


gnrs = ['armchair', 'zigzag']
chkdir = Path('ckpts')

for gnr in gnrs:
   print(f'\n********** {gnr} **********')
   Path.mkdir(chkdir / gnr, exist_ok=True)
   ckpts = chkdir.glob(f'*{gnr}*') 
   for c in ckpts:
      if c.is_file():
         ye = torch.load(c, map_location='cpu')
         epoch = ye['epoch']
         print(f'{c.name} --> {epoch}')
         Path.rename(c, chkdir / gnr / f'epoch-{epoch}.pt')