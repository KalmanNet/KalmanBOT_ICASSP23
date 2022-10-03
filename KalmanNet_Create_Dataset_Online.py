import torch
from KalmanNet_sysmdl import SystemModel
from KalmanNet_data import F,H,T_online,m1_0,m2_0,rdB_data,ratio_data,data_file_specification,N_E_online,N_T_online
from KalmanNet_data import DataGen

# Initialize ssSystem
r = q = 1
ssSystem = SystemModel(F,q,H,r,T_online)
ssSystem.InitSequence(m1_0,m2_0)

print('Start Dataset Creation:')

# Change from dB to absolute value
r = 10 ** (-rdB_data / 20)
q = 10 ** ((ratio_data - rdB_data) / 20)


# Filename for dataset
data_file_name = 'Datasets'+'\\'+ data_file_specification.format(ratio_data,rdB_data,T_online) + '.pt'

print('Training observation noise 1/R^2:',rdB_data,'[dB]')
print('Training ratio:', ratio_data, '[dB]')

ssSystem.UpdateCovariance_Gain(q, r)
DataGen(ssSystem,data_file_name,N_E=N_E_online,N_CV=0,N_T= N_T_online)
