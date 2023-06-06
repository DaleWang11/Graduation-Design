# 考虑直接以完美CSI输入网络，进行混合预编码设计

# 方法：输入直接为CSI，使用三个transformer进行训练
# 结果：se=12.126
import torch
import os
import h5py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from ModelDesign import *
import random
import torch.nn.functional as F
import numpy as np
from math import *
import torch.utils.data as Data
#from torch.utils.data import Dataset
import torch.nn as nn
import datetime
from Model_Trans1 import *

def test(Nc, N, Nt, B, Nr, L, SNR_dB, K, EPOCH, BATCH_SIZE):  # N代表路径数，若N为-1则代表
    shoulian = np.zeros(EPOCH)
    snr = 10 ** (SNR_dB / 10) / K
    parm_set = [Nc, Nt, Nr, snr, B, K]

    # H_train = torch.load('data/H_train_UPA' + str(N) + 'Lp.pt')
    # H_train = H_train[:, 0:K, :, :]
    # print(H_train.shape)
    # H_test = torch.load('data/H_test_UPA' + str(N) + 'Lp.pt')
    # H_test = H_test[:, 0:K, :, :]
    # print(H_test.shape)
    test = 'H_test_'+str(N)+'.mat'
    mat = h5py.File(test)
    H_test = mat['H_test']
    H_test = np.transpose(H_test, [3, 2, 1, 0])
    H_test = H_test.astype('float32')  # 训练变量类型转换
    H_test = H_test[:, 0:K, :, :]

    net_BS = RIS_SDMA_Precoding(parm_set).cuda()

    print(net_BS)

    net_BS = torch.load(
        'Model_driven_test' + str(B) + 'B' + str(N) + 'Ncl' + str(L) + 'L' + str(K) + 'K' + '_8L_UPA_T01_Trans_warm.pth')

    loss_func1 = MyLoss_OFDM()
    loss_func1 = loss_func1.cuda()

    best_SE = 0

    torch_dataset_test = DatasetFolder(H_test)
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练时随机打乱数据
        num_workers=0,  # 每次用两个进程提取
    )
    start = datetime.datetime.now()
    test_SE = 0
    num_test = 0
    net_BS.eval()
    with torch.no_grad():
        for step, b_x in enumerate(loader_test):
            num_test = num_test + 1
            net_BS.eval()  # 验证模式
            b_x = b_x.cuda()
            num = b_x.shape[0]
            out2 = net_BS(b_x,parm_set)
            loss = loss_func1(b_x, out2, parm_set)
            test_SE = test_SE - loss
        test_SE = test_SE / num_test
    time0 = datetime.datetime.now() - start
    print('Epoch:', 'test SE %.3f' % test_SE.cpu())

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

B = 40  # 反馈bit数
N=1

if __name__ == '__main__':
    test(Nc,N,Nt,B,Nr,L,SNR_dB,K,EPOCH,BATCH_SIZE)