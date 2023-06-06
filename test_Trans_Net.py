import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块GPU（从0开始）
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch.nn.functional as F
import torchvision
import numpy as np
from math import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython import display
import torch.utils.data as Data
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# %matplotlib notebook
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *
from trans_1 import *
import h5py
def test(Nc, N, Nt, B, Nr, L, SNR_dB, K, EPOCH, BATCH_SIZE):

    # Parameters for testing
    snr = 10 ** (SNR_dB / 10) / K
    parm_set = [Nc, Nt, Nr, snr, B, K]

    # Data loading for testing
    test = 'H_test_'+str(N)+'.mat'
    mat = h5py.File(test)
    H_test = mat['H_test']
    H_test = np.transpose(H_test, [3, 2, 1, 0])
    H_test = H_test.astype('float32')  # 训练变量类型转换
    H_test = H_test[:, 0:K, :, :]

    #载入保存的已训练好的模型
    net_BS = torch.load(
        './Hatau_' + str(B) + 'B' + str(N) + 'Ncl' + str(L) + 'L' + str(K) + 'K' + '_8L_UPA_T01_Trans_warm.pth')
    # Loss function
    loss_func1 = MyLoss_OFDM()
    loss_func1 = loss_func1.cuda()
    
    #数据载入与数据封装
    torch_dataset_test = DatasetFolder(H_test)
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练时随机打乱数据
        num_workers=0,  # 每次用两个进程提取
    )

    num_test = 0
    test_SE = 0

    with torch.no_grad():
        for step, b_k in enumerate(loader_test):
            num_test = num_test+1
            net_BS.eval()  # HBFN

            b_k = b_k.cuda()
            num = b_k.shape[0]
            out2 = net_BS(b_k, parm_set)
            loss = loss_func1(b_k, out2, parm_set)
            test_SE = test_SE - loss
        test_SE = test_SE / num_test
    print('test SE %.3f' % test_SE.cpu())
    return test_SE.cpu()

Nc = 32
#N = 2  # 多径数
Nt = 64 # 基站端天线数
Nr = 1 # 用户端天线数
# B = 30

L = 8  # 用户端接收的观测，即OFDM个数
SNR_dB = 10
K = 2  # 用户数
snr =  10**(SNR_dB/10)/K

BATCH_SIZE = 512
EPOCH = 180

Bs = [1,3,5,8,16,24,32,48,64]  # 反馈bit数
#N_=[3,4,5,6]
N = 1
if __name__ == '__main__':
    for B in Bs:
        test(Nc,N,Nt,B,Nr,L,SNR_dB,K,EPOCH,BATCH_SIZE)