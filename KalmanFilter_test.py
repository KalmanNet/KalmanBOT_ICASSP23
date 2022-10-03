import numpy as np
import torch.nn
import torch.nn as nn

from KalmanNet_KF import KalmanFilter
from KalmanNet_data import N_T
from tqdm import trange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def KFTest(SysModel,test_dataset):

    # Since the KF is not optimized for Batches, it is best to compute on CPU
    test_input = test_dataset.input.to( torch.device('cpu'))
    test_target = test_dataset.target.to(torch.device('cpu'))

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    # Number of examples
    N_T = test_dataset.input.shape[0]

    # MSE [Linear]
    MSE_KF_linear_arr = np.empty(N_T)
    MSE_KF_linear_arr_obs = np.empty(N_T)


    KF = KalmanFilter(SysModel)


    for j in trange(0, N_T,desc= 'Kalman Filter Test',position = 0, leave = True):

        KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

        KF.GenerateSequence(test_input[j, :, :SysModel.T])

        MSE_KF_linear_arr[j] = loss_fn(KF.x[:,:], test_target[j, :, :SysModel.T]).item()
        MSE_KF_linear_arr_obs[j] = loss_fn(KF.y_pred[:,:],test_input[j,:,:SysModel.T]).item()


    MSE_KF_linear_avg = np.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * np.log10(MSE_KF_linear_avg)

    MSE_KF_linear_avg_obs = np.mean(MSE_KF_linear_arr_obs)
    MSE_KF_dB_avg_obs = 10 * np.log10(MSE_KF_linear_avg_obs)

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter Observation - MSE LOSS", MSE_KF_dB_avg_obs, '[dB]')

    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg,MSE_KF_dB_avg_obs]



