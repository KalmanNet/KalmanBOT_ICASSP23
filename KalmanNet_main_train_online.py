from KalmanNet_data import F,H,T_online,m1_0,m2_0
import torch
import numpy as np
from KalmanNet_sysmdl import SystemModel
from Pipeline_Online import Pipeline_Online
from KalmanFilter_test import KFTest
from KalmanNet_data import DataLoader,data_file_specification,model_file_specification,unsupervised\
    ,N_E_online,N_T_online,rdB_pretrained,rdB_data,ratio_pretrained,ratio_data,training_start,T
from datetime import datetime
from KalmanNet_nn import KalmanNetNN

# Initialize ssModel

ssModel = SystemModel(F,1,H,1,T_online)
ssModel.InitSequence(m1_0,m2_0)


print('Observation Noise pre-trained model 1/r^2:',rdB_pretrained,'[dB]')
print('Ratio pre-trained model q^2/r^2:',ratio_pretrained,'[dB]')

# Load Data
data_file_name = 'Datasets'+'\\' + data_file_specification.format(ratio_data,rdB_data,T_online) + '.pt'
[train_dataset,cv_dataset,test_dataset] = DataLoader(data_file_name)

print('Evaluate Correct Kalman Filter Performance:')
r = 10 ** (-rdB_data / 20)
q = 10 ** ((ratio_data - rdB_data) / 20)
ssModel.UpdateCovariance_Gain(q,r)
[MSE_KF_linear_state,_,MSE_KF_dB_state,MSE_KF_dB_observation] = KFTest(ssModel,test_dataset)

print('Evaluate Wrong Kalman Filter Performance:')
r = 10 ** (-rdB_pretrained / 20)
q = 10 ** ((ratio_pretrained - rdB_pretrained) / 20)
ssModel.UpdateCovariance_Gain(q,r)
[MSE_KF_linear_state,_,MSE_KF_dB_state,MSE_KF_dB_observation] = KFTest(ssModel,test_dataset)

print('Start Pipeline')
Pipeline_name = model_file_specification.format(ratio_data,rdB_data,T_online,unsupervised)
folder_name = 'KNet'
now = datetime.now()
Pipeline = Pipeline_Online(now,folder_name,Pipeline_name)

# Load pre-trained model
pretrained_model_name = model_file_specification.format(ratio_pretrained,rdB_pretrained,T,unsupervised)
model = torch.load('KNet\\model_'+pretrained_model_name+'.pt')

# Set Pipeline Parameters
Pipeline.setssModel(ssModel)
Pipeline.setModel(model)
Pipeline.setTrainingParams(learningRate=1e-4, weightDecay=1e-6, stride = 10, training_start= training_start)

# Test model before online training
print('Model MSE - Before online Training:')
Pipeline.NNTest(n_Test=N_T_online,test_dataset=test_dataset)

# Start Training
training_start = datetime.now()
Pipeline.NNTrain(n_Examples=N_E_online,training_dataset= train_dataset)
training_end = datetime.now()
print('Training took:',training_end-training_start)

# Start Testing
test_start = datetime.now()
print('Model MSE - After online Training')
Pipeline.NNTest(n_Test= N_T_online,test_dataset=test_dataset)
test_end = datetime.now()
print('Testing took:',test_end-test_start)

Pipeline.save()
