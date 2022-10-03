import torch
from KalmanNet_sysmdl import SystemModel
from KalmanNet_data import F,H,T,m1_0,m2_0,R_decibel_train_ranges,ratio,data_file_specification
from KalmanNet_data import DataGen

# Initialize ssSystem
r = q = 1
ssSystem = SystemModel(F,q,H,r,T)
ssSystem.InitSequence(m1_0,m2_0)

print('Start Dataset Creation:')

for nRdB, rdB in enumerate(R_decibel_train_ranges):

    # Change from dB to absolute value
    r = 10 ** (-rdB / 20)
    q = 10 ** ((ratio - rdB) / 20)


    # Filename for dataset
    data_file_name = 'Datasets'+'\\'+ data_file_specification.format(ratio,rdB,T) + '.pt'

    print('Training observation noise 1/R^2:',rdB,'[dB]')
    print('Training ratio:', ratio, '[dB]')

    ssSystem.UpdateCovariance_Gain(q, r)
    DataGen(ssSystem,data_file_name)
