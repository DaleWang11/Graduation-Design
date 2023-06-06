#完美信道输入

import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
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
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *
from Transformer_model import *
import h5py

def Hermitian(X):#torch矩阵共轭转置
    X = torch.real(X) - 1j*torch.imag(X)
    return X.transpose(-1,-2)



class MyLoss_OFDM(torch.nn.Module):  # 输入是信道和整个F,输出是频谱效率
    def __init__(self):
        super(MyLoss_OFDM, self).__init__()

    def forward(self, H0, out, parm_set):  # H0第0个维度是样本 第1个维度是用户，第2个维度是子载波，第3个维度是天线
        Nc = parm_set[0]
        Nt = parm_set[1]
        Nr = parm_set[2]
        snr = parm_set[3]
        B = parm_set[4]
        K = parm_set[5]

        H = H0.permute(0, 2, 1, 3)  # H第0个维度是样本 第1个维度是子载波，第2个维度是用户，第3个维度是天线
        num = out.shape[0]
        Nc = H.shape[1]
        H_real = H[:, :, :, 0:Nt]
        H_imag = H[:, :, :, Nt:2 * Nt]
        Hs = torch.zeros([num, Nc, K, Nt * 2])
        Hs = Hs.cuda()
        Hs[:, :, 0:K, 0:Nt] = H_real
        # Hs[:, :, K:2 * K, Nt:2 * Nt] = H_real
        Hs[:, :, 0:K, Nt:2 * Nt] = H_imag
        # Hs[:, :, K:2 * K, 0:Nt] = -H_imag

        F = torch.zeros([num, Nc, Nt * 2, K * 2])
        F = F.cuda()
        F[:, :, 0:Nt, 0:K] = out[:, 0:K * Nt * Nc].reshape(num, Nc, Nt, K)
        F[:, :, Nt:2 * Nt, K:2 * K] = out[:, 0:K * Nt * Nc].reshape(num, Nc, Nt, K)
        F[:, :, 0:Nt, K:2 * K] = out[:, K * Nt * Nc:2 * K * Nt * Nc].reshape(num, Nc, Nt, K)
        F[:, :, Nt:2 * Nt, 0:K] = -out[:, K * Nt * Nc:2 * K * Nt * Nc].reshape(num, Nc, Nt, K)
        R = 0
        Hk = torch.matmul(Hs, F)
        noise = 1 / snr
        for i in range(K):
            signal = Hk[:, :, i, i] * Hk[:, :, i, i] + Hk[:, :, i, i + K] * Hk[:, :, i, i + K]
            interference = torch.zeros(num, Nc)
            interference = interference.cuda()
            for j in range(K):
                if j != i:
                    interference = interference + Hk[:, :, i, j] * Hk[:, :, i, j] + Hk[:, :, i, j + K] * Hk[:, :, i,
                                                                                                         j + K]
            SINR = signal / (noise + interference)
            R = R + torch.sum(torch.log2(1 + SINR))
        R = -R / num / Nc
        return R
    
class DatasetFolder(Data.Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]

class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class RIS_SDMA_Precoding(nn.Module): #单层
    def __init__(self, parm_set): 
        super(RIS_SDMA_Precoding,self).__init__()
        
        Nc = parm_set[0]
        Nt = parm_set[1]
        Nr = parm_set[2]
        snr = parm_set[3]
        B = parm_set[4]
        K = parm_set[5]
        
        self.trans_block1 = TRANS_BLOCK(2*Nt*K,2*Nt*K,512,6)
        self.trans_block2 = TRANS_BLOCK(2*Nt*Nc,Nt,128,3)
        
        #self.linear_H  = nn.Linear(Nt*2, 32)
        
        #self.trans1  = TRANS_BLOCK(32*K,32,256,3)
        #self.linear_RIS = nn.Linear(Nc*32, Nt)   #生成RIS相位

        #self.linear_RF = nn.Linear(Nc*32, Nt*K)   #生成RIS相位
        
        self.trans2  = TRANS_BLOCK(2*K*K,4*K+1,256,1) #4×4等效信道
        
    
    def forward(self, H):
        # H(batch,K,Nc,Nt)
        #H_RU [batch,K,Nc,1,M_ant]  s [batch,Nc,2*K] H_BR[1,Nc,M_ant,M_ant]
        # param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L,SNR]
        #Nc = parm_set[0]
        #Nt = parm_set[1]
        #Nr = parm_set[2]
        #snr = parm_set[3]
        #B = parm_set[4]
        #K = parm_set[5]
        
        batch = H.shape[0]
        '''
        h = H.permute(0,2,1,3)
        x = h.contiguous().view(-1,int(Nc),int(2*Nt*K))

        #H [batch,K,Nc,2*Nt]
        x_ = self.trans_block1(x)
        x_ = x.contiguous().view(-1, Nc, int(Nt*2), K)
        x_ = x_.permute(0,3,1,2)
        x_ = x_.reshape(-1,K,Nc*Nt*2)
        x_ = self.trans_block2(x_)
        x_ = x_.permute(0,2,1) #(-1,Nt,K)

        F_RF = (torch.zeros(batch,Nt,K)+ 0j).cuda()

        RF_real = torch.cos(x_).reshape(-1, Nt, K) / sqrt(Nt)
        RF_imag = torch.sin(x_).reshape(-1, Nt, K) / sqrt(Nt)

        F_RF = torch.complex(RF_real,RF_imag)
        '''


        #x = self.trans1(x)   #[batch,Nc,32]
        #out_RIS = x.reshape(-1,Nc*32)
        #Phi_phase = self.linear_RIS(out_RIS)
        #Phi = torch.exp(1j*Phi_phase.reshape(-1,1,M_ant))
        #Phi = torch.diag_embed(Phi)  #[batch,1,M_ant,M_ant]
        '''
        out_RF = x[:,:,32:64].reshape(-1,Nc*32)
        RF_phase = self.linear_RF(out_RF)
        F_RF = torch.exp(1j*RF_phase.reshape(-1,Nt,K))/sqrt(Nt) #[batch,1,M_ant,K]
        '''
        #H[batch,K,Nc,Nt]
        H1 = (torch.zeros(batch,K,Nc,Nt)+ 0j).cuda()
        H1 = torch.complex(H[:,:,:,0:Nt],H[:,:,:,:Nt:2*Nt])     #合并完成为复数矩阵
        
        F_RF = (torch.zeros(batch,Nt,K)+ 0j).cuda()
        #不同用户的模拟预编码矩阵不同
        for i in range(K):
            F_RF[:,:,i] = H1[:,i,Nc//2,:]
        F_RF = F_RF.reshape(batch,Nt,K)
        F_RF = torch.real(F_RF) - 1j*torch.imag(F_RF)        #共轭转置
        F_RF = (F_RF/torch.abs(F_RF))/sqrt(Nt)
        
        #print(H1.size())
        #print(F_RF.size())
        H_equ = (torch.zeros(batch,Nc,K,K) + 0j).cuda()
        H1 = H1.permute(2,0,1,3)
        #H1 = H1.reshape([Nc,batch,K,Nt])
        H_equ = (H1 @ F_RF).permute(1,0,2,3)    #(batch,K,Nc,K)
           
        n = (torch.randn(H.shape[0],Nc,K) + 1j*torch.randn(H.shape[0],Nc,K)).cuda()/sqrt(2 * snr)   #噪声
        
        power = torch.sum(torch.abs(H_equ*H_equ),[2,3])
        H_equ = H_equ/torch.sqrt(power.reshape(-1,Nc,1,1))*sqrt(K)
        

        H_equ1 = H_equ.reshape(-1,Nc,K*K)
        x2 = torch.cat((torch.real(H_equ1),torch.imag(H_equ1)), 2) #[batch,Nc,2*K*K]
        x3 = self.trans2(x2)   #[batch,Nc,4*K+1]
    
        H_hat = H_equ.to(torch.complex128)
        pri1 = (x3[:,:,0:K] + 1j*x3[:,:,K:2*K]).to(torch.complex128)    #[batch,Nc,K]
        pri2 = (x3[:,:,2*K:3*K] + 1j*x3[:,:,3*K:4*K]).to(torch.complex128)   #[batch,Nc,K]
        # pub1 = (x[:,:,4*K:5*K] + 1j*x[:,:,5*K:6*K]).to(torch.complex128) #[batch,Nc,K]
        # pub2 = (x[:,:,6*K:7*K] + 1j*x[:,:,7*K:8*K]).to(torch.complex128) #[batch,Nc,K]

        sigma_pri = (x3[:,:,4*K]).to(torch.complex128)  #[batch,Nc]
        # sigma_pub = (x[:,:,8*K+1]).to(torch.complex128)#[batch,Nc]


        for k in range(K):
            hk = Hermitian(H_hat[:,:,k,:].reshape(batch,Nc,1,K))
            if(k==0):
                B_pri = pri2[:,:,k].reshape(batch,Nc,1,1) * hk @ Hermitian(hk)
                B_pri = B_pri + (sigma_pri.reshape(batch,Nc,1,1) * torch.eye(K).cuda().to(torch.complex128))

                # B_pub = pub2[:,:,k].reshape(batch,Nc,1,1) * hk @ Hermitian(hk)
                # B_pub = B_pub + (sigma_pub.reshape(batch,Nc,1,1) * torch.eye(K).cuda().to(torch.complex128))
            else:
                B_pri = B_pri + pri2[:,:,k].reshape(batch,Nc,1,1) * hk @ Hermitian(hk)
                B_pri = B_pri + (sigma_pri.reshape(batch,Nc,1,1) * torch.eye(K).cuda().to(torch.complex128))

                # B_pub = B_pub + pub2[:,:,k].reshape(batch,Nc,1,1) * hk @ Hermitian(hk)
                # B_pub = B_pub + (sigma_pub.reshape(batch,Nc,1,1) * torch.eye(K).cuda().to(torch.complex128))

        V_pri = (torch.zeros((batch,Nc,K,K)) + 0j).cuda().to(torch.complex128)
        # V_pub = (torch.zeros((batch,Nc,K,1)) + 0j).cuda().to(torch.complex128)
        # B_inv_pub = torch.inverse(B_pub)
        B_inv_pri = torch.inverse(B_pri)
        # print(np.sum(np.abs(A_inv)**2))
        for k in range(K):
            hk = Hermitian(H_hat[:,:,k,:].reshape(batch,Nc,1,K))
            V_pri[:,:,:,k] = (B_inv_pri @ (pri1[:,:,k].reshape(batch,Nc,1,1) * hk)).reshape(batch,Nc,K)
        # for k in range(K):
        #     hk = Hermitian(H_hat[:,:,k,:].reshape(batch,Nc,1,K))
        #     if k==0:
        #         A = pub1[:,:,k].reshape(batch,Nc,1,1) * hk
        #     else:
        #         A = A + pub1[:,:,k].reshape(batch,Nc,1,1) * hk
        # V_pub = (B_inv_pub @ A)

        w_pub = (torch.zeros((batch,Nc,K,1)) + 0j).cuda().to(torch.complex128)  #没有public signal
        W_pri = V_pri


        F_BB_RS = (w_pub + 0).to(torch.complex64)

        F_BB_SDMA = (W_pri + 0).to(torch.complex64)   #[batch,Nc,K,K]
        F_BB_SDMA = F_BB_SDMA.reshape([Nc,batch,K,K]) # digital precoding
        F_SDMA = (F_RF @ F_BB_SDMA).to(torch.complex64)  #[batch,Nc,M_ant,K]         [batch,Nt,K] * [Nc,batch,K,K] = [Nc,batch,Nt,K]广播机制
        #F_RS = F_RF @ F_BB_RS  #[batch,Nc,M_ant,N_RS1]
        F_SDMA = F_SDMA.reshape([batch,Nc,Nt,K])
        Power = (torch.sum(torch.abs(F_SDMA)**2,[2,3])).reshape(-1,Nc,1,1)   #[batch,Nc,1,1]  [batch,Nc,Nt,K]
        #Power1 = (torch.sum(torch.abs(F_SDMA)**2,[1,2,3])).reshape(-1,1,1,1)
        #print(Power1.size())
        #print(F_BB_SDMA.size())
        F_BB_SDMA = F_BB_SDMA.permute(1,0,2,3)
        #F_SDMA = F_SDMA / torch.sqrt(Power) * sqrt(K)
        F_BB_SDMA = F_BB_SDMA / torch.sqrt(Power) * sqrt(K)  #F_BB做归一化 由于吴总保证SNR的原因要乘以sqrt(K)
        #F_BB_RS = F_BB_RS / torch.sqrt(Power) * sqrt(K)
        #归一化之后的预编码矩阵
        
        F_BB_SDMA = F_BB_SDMA.permute(1,0,2,3)
        F_SDMA = F_RF @ F_BB_SDMA
        F_SDMA = F_SDMA.permute(1,0,2,3)
        F_SDMA = F_SDMA.reshape([batch,K*Nc*Nt])

    
        #abs(A)**2
        F_real = torch.real(F_SDMA).reshape(batch, K * Nt * Nc)
        F_imag = torch.imag(F_SDMA).reshape(batch, K * Nt * Nc)
        F = torch.cat((F_real, F_imag), 1)
        return F  #Phi[batch,1,M_ant,M_ant]  F_RF[batch,1,M_ant,K]  F_BB_SDMA[batch,Nc,K,K] F_BB_RS[batch,Nc,K,1] 
#         print(H_RU.shape)

#正常做法都是F_RF除以Nt,就把他想到和码本一样，F_BB满足功率约束，看后面功率是多少定如何归一化







def train(Nc, N, Nt, B, Nr, L, SNR_dB, K, EPOCH, BATCH_SIZE):  # N代表路径数，若N为-1则代表
    shoulian = np.zeros(EPOCH)
    snr = 10 ** (SNR_dB / 10) / K
    parm_set = [Nc, Nt, Nr, snr, B, K]

    # H_train = torch.load('data/H_train_UPA' + str(N) + 'Lp.pt')
    # H_train = H_train[:, 0:K, :, :]
    # print(H_train.shape)
    # H_test = torch.load('data/H_test_UPA' + str(N) + 'Lp.pt')
    # H_test = H_test[:, 0:K, :, :]
    # print(H_test.shape)
    train = 'H_train_'+str(N)+'.mat'
    mat = h5py.File(train)
    H_train = mat['H_train']
    H_train = np.transpose(H_train, [3, 2, 1, 0])
    H_train = H_train.astype('float32')  # 训练变量类型转换
    print(H_train.shape)
    test = 'H_val_'+str(N)+'.mat'
    mat = h5py.File(test)
    H_test = mat['H_val']
    H_test = np.transpose(H_test, [3, 2, 1, 0])
    H_test = H_test.astype('float32')  # 训练变量类型转换

    net_BS = RIS_SDMA_Precoding(parm_set).cuda()

    print(net_BS)
    opt = NoamOpt(256, 1, 4000, torch.optim.Adam(net_BS.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    #optimizer_BS = torch.optim.Adam(net_BS.parameters(), lr=0.001)
    #scheduler_BS = torch.optim.lr_scheduler.MultiStepLR(optimizer_BS, milestones=[100, 150], gamma=0.6, last_epoch=-1)

    loss_func1 = MyLoss_OFDM()
    loss_func1 = loss_func1.cuda()

    best_SE = 0
    torch_dataset_train = DatasetFolder(H_train)
    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练时随机打乱数据
        num_workers=0,  # 每次用两个进程提取
    )

    torch_dataset_test = DatasetFolder(H_test)
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练时随机打乱数据
        num_workers=0,  # 每次用两个进程提取
    )
    start = datetime.datetime.now()
    for epoch in range(EPOCH):
        print('========================')
        print('lr:%.4e' % opt.optimizer.param_groups[0]['lr'])
        train_SE = 0
        num_train = 0
        test_SE = 0
        num_test = 0
        for step, b_x in enumerate(loader_train):
            num_train = num_train + 1
            net_BS.train()  # 训练模式
            b_x = b_x.cuda()
            num = b_x.shape[0]
            out2 = net_BS(b_x, parm_set)
            loss = loss_func1(b_x, out2, parm_set)
            train_SE = train_SE - loss
            opt.optimizer.zero_grad()

            loss.backward()
            opt.step()
        train_SE = train_SE / num_train
        # scheduler_BS.step()
        net_BS.eval()
        with torch.no_grad():
            for step, b_x in enumerate(loader_test):
                num_test = num_test + 1
                net_BS.eval()  # 验证模式
                b_x = b_x.cuda()
                num = b_x.shape[0]
                out2 = net_BS(b_x, parm_set)
                loss = loss_func1(b_x, out2, parm_set)
                test_SE = test_SE - loss
            test_SE = test_SE / num_test

        time0 = datetime.datetime.now() - start
        print('Epoch:', epoch, 'time', time0, 'train SE %.3f' % train_SE.cpu(), 'test SE %.3f' % test_SE.cpu())
        start = datetime.datetime.now()

        if test_SE > best_SE:
            best_SE = test_SE
            torch.save(net_BS,
                       './Model_driven_test' + str(B) + 'B' + str(N) + 'Ncl' + str(L) + 'L' + str(K) + 'K' + '_8L_UPA_T01_Trans_warm.pth')
            print('Model saved!')
        shoulian[epoch] = test_SE.cpu()
    print('The best SE is: %.3f' % best_SE.cpu())
    print(shoulian)

Nc = 32
# N = 2  # 多径数
Nt = 64 # 基站端天线数
Nr = 1 # 用户端天线数
# B = 30

L = 8  # 用户端接收的观测，即OFDM个数
SNR_dB = 10
K = 2  # 用户数
snr = 10**(SNR_dB/10)/K

BATCH_SIZE = 512
EPOCH = 180

B = 40  # 反馈bit数
#N_=[5,6]
N=1

if __name__ == '__main__':
    train(Nc,N,Nt,B,Nr,L,SNR_dB,K,EPOCH,BATCH_SIZE)