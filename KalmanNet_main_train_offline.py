from KalmanNet_data import F,H,T,m1_0,m2_0
import torch
import numpy as np
from KalmanNet_sysmdl import SystemModel
from Pipeline_Offline import Pipeline_Offline
from KalmanFilter_test import KFTest
from KalmanNet_data import DataGen,DataLoader,R_decibel_train_ranges,ratio,data_file_specification,\
    model_file_specification,unsupervised,N_E,N_T,N_CV
from datetime import datetime
from KalmanNet_nn import KalmanNetNN

# Initialize ssModel

ssModel = SystemModel(F,1,H,1,T)
ssModel.InitSequence(m1_0,m2_0)

# Start going through the training range
for nrdB, rdB in enumerate(R_decibel_train_ranges):

    print('Observation Noise 1/r^2:',rdB,'[dB]')
    print('Ratio q^2/r^2:',ratio,'[dB]')

    # Update ssModel
    r = 10 ** (-rdB / 20)
    q = 10 ** ((ratio - rdB) / 20)
    ssModel.UpdateCovariance_Gain(q,r)

    # Load Data
    data_file_name = 'Datasets'+'\\' + data_file_specification.format(ratio,rdB,T) + '.pt'
    [train_dataset,cv_dataset,test_dataset] = DataLoader(data_file_name)

    print('Evaluate Kalman Filter Performance:')
    [MSE_KF_linear_state,_,MSE_KF_dB_state,MSE_KF_dB_observation] = KFTest(ssModel,test_dataset)

    print('Start Pipeline')
    model_name = model_file_specification.format(ratio,rdB,T,unsupervised)
    folder_name = 'KNet'
    now = datetime.now()
    Pipeline = Pipeline_Offline(now,folder_name,model_name)

    # Build NN
    model = KalmanNetNN()
    model.Build(ssModel)

    # Set Pipeline Parameters
    Pipeline.setssModel(ssModel)
    Pipeline.setModel(model)
    Pipeline.setTrainingParams(n_Epochs= 500, n_Batch= 50, learningRate=1e-3, weightDecay=1e-6, unsupervised= unsupervised)

    # Start Training
    training_start = datetime.now()
    Pipeline.NNTrain(n_Examples=N_E,training_dataset= train_dataset,n_CV= N_CV, cv_dataset= cv_dataset)
    training_end = datetime.now()
    print('Training took:',training_end-training_start)

    # Start Testing
    test_start = datetime.now()
    Pipeline.NNTest(n_Test= N_T,test_dataset=test_dataset)
    test_end = datetime.now()
    print('Testing took:',test_end-test_start)

    Pipeline.save()
    Pipeline.PlotTrain(MSE_KF_linear_state,MSE_KF_dB_state)