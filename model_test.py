# =======================================================================================================================
# =======================================================================================================================
import pickle
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import h5py
import time
import os
import math
from modelDesign import *


# Parameters for training
gpu_list = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
# =======================================================================================================================
# =======================================================================================================================
# Parameters Setting for Data
CHANNEL_SHAPE_DIM1 = 32
CHANNEL_SHAPE_DIM2 = int(8*8)
CHANNEL_SHAPE_DIM3 = 2

# Parameters Setting for Training
BATCH_SIZE = 128
EPOCHS = 50
num_workers = 0  # 2
LEARNING_RATE = 1e-4
PRINT_RREQ = 250
# torch.manual_seed(1)
# =======================================================================================================================
# =======================================================================================================================
# Data Loading


test = 'H_test_6.mat'


mat = h5py.File(test)
#mat = sio.loadmat(test)
data_test = mat['H_test']
data_test = np.transpose(data_test, [3, 2, 1, 0])
data_test = data_test.astype('float32')  # 训练变量类型转换

test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
# =======================================================================================================================
# =======================================================================================================================
BIT_List = [64,128,256,512,1024]
saveresultsname = 'NMSE_Q_CSI_Feedback_Transformer.mat'
NMSE = np.zeros(dtype=np.float32, shape=(len(BIT_List), 1))
D = dict(nmse=NMSE)
sio.savemat(saveresultsname, D)
results = sio.loadmat(saveresultsname)
for ind in range(len(BIT_List)):
    NUM_FEEDBACK_BITS = BIT_List[ind]
    nmse = []

    # Model Constructing
    autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS,CHANNEL_SHAPE_DIM1,CHANNEL_SHAPE_DIM2,CHANNEL_SHAPE_DIM3)

    autoencoderModel = autoencoderModel.cuda()
    model_encoder = autoencoderModel.encoder
    model_encoder.load_state_dict(torch.load('./models/encoder_q_384_'+str(NUM_FEEDBACK_BITS)+'.pth.tar')['state_dict'])
    model_decoder = autoencoderModel.decoder
    model_decoder.load_state_dict(torch.load('./models/decoder_q_384_'+str(NUM_FEEDBACK_BITS)+'.pth.tar')['state_dict'])

    #summary(autoencoderModel, input_size=(32,64*2))
    #criterion = nn.MSELoss().cuda()
    criterion_test = NMSELoss(reduction='mean')
    # =======================================================================================================================
    # =======================================================================================================================
    # Model Testing
    t1 = time.time()
    print('========================')
    autoencoderModel.eval()
    totalLoss = 0
    with torch.no_grad():
        for i, autoencoderInput in enumerate(test_loader):
            autoencoderInput = autoencoderInput.cuda()
            autoencoderOutput = autoencoderModel(autoencoderInput)
            loss = criterion_test(autoencoderInput, autoencoderOutput)
            totalLoss += loss.item()
        averageLoss = totalLoss / len(test_loader)
        dB_loss = 10*math.log10(averageLoss)
        print('Loss %.4f' % averageLoss, 'dB Loss %.4f' % dB_loss)
        nmse = np.append(nmse, dB_loss)
        NMSE = results['nmse']
        NMSE[ind, :] = nmse
        D = dict(nmse=NMSE)
        sio.savemat(saveresultsname, D)
    t2 = time.time()
    print('Time: ', t2-t1)
    print('Testing for '+str(NUM_FEEDBACK_BITS)+' bits is finished!')
    # =======================================================================================================================
    # =======================================================================================================================
