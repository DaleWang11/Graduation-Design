# 考虑直接以完美CSI输入网络，进行混合预编码设计

import torch
import os
import h5py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二块GPU（从0开始）
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


def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 0].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        #b, c = grad_output.shape
        #grad_bit = grad_output.repeat(1, 1, ctx.constant)
        grad_bit = grad_output.repeat_interleave(ctx.constant,dim=1)
        return grad_bit, None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out

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


def mish(x):
    x = x * (torch.tanh(F.softplus(x)))
    return x


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    #         print("Mish activation loaded...")
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class RES_BLOCK(nn.Module):  # 输入信道输出 量化后的B比特反馈信息
    def __init__(self, channel_list):
        super(RES_BLOCK, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_list[0],  # 图片通道数
                out_channels=channel_list[1],  # filter数量
                kernel_size=(5, 1),  # filter大小
                stride=1,  # 扫描步进
                padding=(2, 0),  # 周围围上2圈0 使得输出的宽和高和之前一样不变小
            ),
            nn.BatchNorm2d(channel_list[1], eps=1e-05, momentum=0.1, affine=True),
            Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_list[1],  # 图片通道数
                out_channels=channel_list[2],  # filter数量
                kernel_size=(5, 1),  # filter大小
                stride=1,  # 扫描步进
                padding=(2, 0),  # 周围围上2圈0 使得输出的宽和高和之前一样不变小
            ),
            nn.BatchNorm2d(channel_list[2], eps=1e-05, momentum=0.1, affine=True),
            Mish(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_list[2],  # 图片通道数
                out_channels=channel_list[0],  # filter数量
                kernel_size=(5, 1),  # filter大小
                stride=1,  # 扫描步进
                padding=(2, 0),  # 周围围上2圈0 使得输出的宽和高和之前一样不变小
            ),
            nn.BatchNorm2d(channel_list[0], eps=1e-05, momentum=0.1, affine=True),
        )

    def forward(self, x_ini):
        x = self.conv1(x_ini)
        x = self.conv2(x)
        x = self.conv3(x)
        x = mish(x + x_ini)
        return x


def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(- 2 * pi * 1J / N)
    W = np.power(omega, i * j) / sqrt(N)
    return np.mat(W)

def Hermitian(X):#torch矩阵共轭转置
    X = torch.real(X) - 1j*torch.imag(X)
    return X.transpose(-1,-2)  #（3,4）变（4,3）

Nc = 32
Nt = 64
W = DFT_matrix(Nc)
W = torch.tensor(W)
W = Hermitian(W)
W_real = np.real(W).cuda()
W_imag = np.imag(W).cuda()
W_real = W_real.float()
W_imag = W_imag.float()

W1 = DFT_matrix(Nt)
W_real1 = torch.from_numpy(np.real(W1)).cuda()
W_imag1 = torch.from_numpy(np.imag(W1)).cuda()
W_real1 = W_real1.float()
W_imag1 = W_imag1.float()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class DNN_BS_hyb_OFDM(nn.Module):  # 输入所有用户的KB比特反馈信息,输出预编码
    def __init__(self, parm_set):
        Nc = parm_set[0]
        Nt = parm_set[1]
        Nr = parm_set[2]
        snr = parm_set[3]
        B = parm_set[4]
        K = parm_set[5]
        super(DNN_BS_hyb_OFDM, self).__init__()


        self.conv_l1 = conv3x3(2,16)
        self.conv_bn_l1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True)
        self.conv_mish_l1 = Mish()

        self.conv_l2 = conv3x3(16,2)
        self.conv_bn_l2 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.conv_mish_l2 = Mish()

        self.FC_l1 = nn.Linear(K*Nc * 2 * Nt, 1024)  # 全连接
        self.bn_l1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()

        self.FC_l2 = nn.Linear(1024, 512)  # 全连接
        self.bn_l2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.FC_l3 = nn.Linear(512, K*B)  # 全连接
        self.bn_l3 = nn.BatchNorm1d(K*B)
        self.QL = QuantizationLayer(1)  # 1bit 量化


        self.DQL = DequantizationLayer(1)  # 解1bit 量化

        self.FC1 = nn.Linear(K * B, Nc*K*Nt*2)  # 全连接
        self.bn1 = nn.BatchNorm1d(Nc*K*Nt*2)
        self.mish1 = Mish()

        # self.FC2 = nn.Linear(2048, 1024)  # 全连接
        # self.bn2 = nn.BatchNorm1d(1024)
        # self.mish2 = Mish()
        #
        # self.FC3 = nn.Linear(1024, 2 * K * K * Nc + K * Nt)  # 全连接
        # self.bn3 = nn.BatchNorm1d(2 * K * K * Nc)
        # self.mish3 = Mish()
        #
        # self.res1 = RES_BLOCK([2 * K * K, 256, 512])
        # self.conv = nn.Conv2d(
        #     in_channels=2 * K * K,  # 图片通道数
        #     out_channels=2 * K * K,  # filter数量
        #     kernel_size=(5, 1),  # filter大小
        #     stride=1,  # 扫描步进
        #     padding=(2, 0),  # 周围围上2圈0 使得输出的宽和高和之前一样不变小
        # )
        # 最后用sigm

        src_vocab_size_0 = int(2*Nt*K)
        model_dimension_0 = 128
        dropout_probability_0 = 0.1
        number_of_heads_0 = 8
        log_attention_weights_0 = False
        number_of_layers_0 = 3
        self.src_embedding_0 = Embedding(src_vocab_size_0, model_dimension_0)  # 对输入进行embedding
        self.src_pos_embedding_0 = PositionalEncoding(model_dimension_0, dropout_probability_0)
        mha_0 = MultiHeadedAttention(model_dimension_0, number_of_heads_0, dropout_probability_0, log_attention_weights_0)
        pwn_0 = PositionwiseFeedForwardNet(model_dimension_0, dropout_probability_0)
        encoder_layer_0 = EncoderLayer(model_dimension_0, dropout_probability_0, mha_0, pwn_0)
        self.trans_encoder_0 = Trans_Encoder(encoder_layer_0, number_of_layers_0, src_vocab_size_0)

        # self.FC3 = nn.Linear(K*Nc*2, 2 * K * K * Nc + K * Nt)  # 全连接
        # self.bn3 = nn.BatchNorm1d(2 * K * K * Nc)
        # self.mish3 = Mish()
        src_vocab_size_1 = int(2*Nt*Nc)
        model_dimension_1 = 128
        dropout_probability_1 = 0.1
        number_of_heads_1 = 8
        log_attention_weights_1 = False
        number_of_layers_1 = 3
        self.src_embedding_1 = Embedding(src_vocab_size_1, model_dimension_1)  # 对输入进行embedding
        self.src_pos_embedding_1 = PositionalEncoding(model_dimension_1, dropout_probability_1)
        mha_1 = MultiHeadedAttention(model_dimension_1, number_of_heads_1, dropout_probability_1, log_attention_weights_1)
        pwn_1 = PositionwiseFeedForwardNet(model_dimension_1, dropout_probability_1)
        encoder_layer_1 = EncoderLayer(model_dimension_1, dropout_probability_1, mha_1, pwn_1)
        self.trans_encoder_1 = Trans_Encoder(encoder_layer_1, number_of_layers_1, Nt)

        src_vocab_size = int(2*Nt*K)
        model_dimension = 128
        dropout_probability = 0.1
        number_of_heads = 8
        log_attention_weights = False
        number_of_layers = 3
        self.src_embedding = Embedding(src_vocab_size, model_dimension)  # 对输入进行embedding
        self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        self.trans_encoder = Trans_Encoder(encoder_layer, number_of_layers, 2*K*K)

    def forward(self, h):
       
        
        #将信道从空间频率域变到角度延时域
        h_real = h[:, :, :, 0:Nt].reshape(-1, K, Nc, Nt, 1)
        h_imag = h[:, :, :, Nt:2 * Nt].reshape(-1, K, Nc, Nt, 1)

        h_real1 = (torch.matmul(W_real1, h_real) - torch.matmul(W_imag1, h_imag)).reshape(-1, 1, K, Nc, Nt)
        h_imag1 = (torch.matmul(W_real1, h_imag) + torch.matmul(W_imag1, h_real)).reshape(-1, 1, K, Nc, Nt)
        #h_sum1 = torch.cat((h_real1, h_imag1), 1)

        h_real2 = (torch.matmul(W_real, h_real1) - torch.matmul(W_imag, h_imag1)).reshape(-1, 1, K, Nc, Nt)
        h_imag2 = (torch.matmul(W_real, h_imag1) + torch.matmul(W_imag, h_real1)).reshape(-1, 1, K, Nc, Nt)
        h_sum2 = torch.cat((h_real2, h_imag2), 1)   #(BATCH_SIZE,2,K,Nc,Nt)

        h = h_sum2.reshape(-1,K,Nc,2,Nt)

        h = h.permute(0,3,1,2,4)
        x0 = h.contiguous().view(-1,2,Nc*K,Nt)

        x = self.conv_l1(x0)
        x = self.conv_bn_l1(x)
        x = self.conv_mish_l1(x)

        x = self.conv_l2(x)
        x = self.conv_bn_l2(x)
        x = self.conv_mish_l2(x+x0)

        x = x.contiguous().view(-1,2*Nc*K*Nt)

        x = self.FC_l1(x)
        x = self.bn_l1(x)
        x = self.relu1(x)
        x = self.FC_l2(x)
        x = self.bn_l2(x)
        x = self.relu2(x)
        x = self.FC_l3(x)
        x = self.bn_l3(x)
        x = torch.sigmoid(x)
        x = self.QL(x)

        x = self.DQL(x) - 0.5

        x = x.contiguous().view(-1,int(K*B))
        x = self.FC1(x)
        x = self.bn1(x)
        x = self.mish1(x)

        x = x.contiguous().view(-1,int(Nc),int(2*Nt*K))
        x = self.src_embedding_0(x)  # get embedding vectors for src token ids
        x = self.src_pos_embedding_0(x)  # add positional embedding
        x = self.trans_encoder_0(x, src_mask=None)  # forward pass through the
        x_ = x.contiguous().view(-1, Nc, int(Nt*2), K)
        x_ = x_.permute(0,3,1,2)
        x_ = x_.reshape(-1,K,Nc*Nt*2)
        x_ = self.src_embedding_1(x_)  # get embedding vectors for src token ids
        x_ = self.src_pos_embedding_1(x_)  # add positional embedding
        x_ = self.trans_encoder_1(x_, src_mask=None)  # forward pass through the
        x_ = x_.permute(0,2,1)
        # x = self.FC1(x)
        # x = self.bn1(x)
        # x = self.mish1(x)
        #
        # x = self.FC2(x)
        # x = self.bn2(x)
        # x = self.mish2(x)

        #x_ini = self.FC3(x)

        RF_real = torch.cos(x_).reshape(-1, Nt, K) / sqrt(Nt)
        RF_imag = torch.sin(x_).reshape(-1, Nt, K) / sqrt(Nt)

        # x = x_ini[:, 0:2 * K * K * Nc]
        # x = self.bn3(x)
        # x = self.mish3(x)

        #x = x.reshape(-1, Nc, 2 * K * K)
        src_embeddings_batch = self.src_embedding(x)  # get embedding vectors for src token ids
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)  # add positional embedding
        x = self.trans_encoder(src_embeddings_batch, src_mask=None)  # forward pass through the encoder
        # x = self.res1(x)
        # x = self.conv(x)

        # x = x.transpose(1, 2)

        BB_real = x[:, :, 0:K * K].reshape(-1, Nc, K, K)
        BB_imag = x[:, :, K * K:2 * K * K].reshape(-1, Nc, K, K)
        BB_real = BB_real.permute(1, 0, 2, 3)
        BB_imag = BB_imag.permute(1, 0, 2, 3)

        F_real = (torch.matmul(RF_real, BB_real) - torch.matmul(RF_imag, BB_imag)).reshape(Nc, -1, K * Nt)
        F_imag = (torch.matmul(RF_imag, BB_real) + torch.matmul(RF_imag, BB_real)).reshape(Nc, -1, K * Nt)
        F_real = F_real.permute(1, 0, 2)
        F_imag = F_imag.permute(1, 0, 2)
        F = torch.cat((F_real, F_imag), 2)

        F_sigma = torch.sqrt(torch.sum(F * F, [2]))
        sigma2 = torch.FloatTensor([sqrt(K)]).cuda()
        F = F / F_sigma.reshape(-1, Nc, 1) * torch.min(F_sigma, sigma2).reshape(-1, Nc, 1)

        F_real = F[:, :, 0:K * Nt].reshape(-1, K * Nt * Nc)
        F_imag = F[:, :, K * Nt:2 * K * Nt].reshape(-1, K * Nt * Nc)
        F = torch.cat((F_real, F_imag), 1)

        return F

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

    net_BS = DNN_BS_hyb_OFDM(parm_set).cuda()

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
                       './Hatau_' + str(B) + 'B' + str(N) + 'Ncl' + str(L) + 'L' + str(K) + 'K' + '_8L_UPA_T01_Trans_warm.pth')
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
Bs = [1,3,5,8,16,24,32,48,64]
if __name__ == '__main__':
    for B in Bs:
        train(Nc,N,Nt,B,Nr,L,SNR_dB,K,EPOCH,BATCH_SIZE)