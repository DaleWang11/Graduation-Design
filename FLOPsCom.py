import torchvision.models as models
import torch
from Model_Trans1 import *
from trans_1 import *
from ptflops import get_model_complexity_info

Nc = 32
Nt = 64
Nr = 1
SNR_dB = 10
snr = 10**(SNR_dB/10)/K
B = 40
K = 2
parm_set = [Nc,Nt,Nr,snr,B,K]

with torch.cuda.device(0):
  net = RIS_SDMA_Precoding(parm_set).cuda()
  macs, params = get_model_complexity_info(net, ((2,32, 64)), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)

  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))