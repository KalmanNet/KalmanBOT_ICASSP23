import torch
import math
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#######################
### Size of DataSet ###
#######################

# Number of Training Examples
N_E = 10_0

# Number of Cross Validation Examples
N_CV = 100

# Number of Testing Examples
N_T = 10_0

# Select space dimensions
############
## 2 x 2 ###
############
m = 2
n = 2

# m = 5
# n = 5

# m = 10
# n = 10

# Initial Conditions
m1x_0_design = torch.ones((m,1))
m2x_0_design = 1 * 1 * torch.eye(m)

# Create canonical F
F = torch.eye(m)
F[0,:] = 1

# Create inverse canonical H
H = torch.zeros((n,m))
H[0,:] = 1
for i in range(n):
    for j in range(m):
        if j == m-i-1:
            H[i,j] = 1


m1_0 = m1x_0_design
m2_0 = m2x_0_design

########################################################################################################################
### For offline use case
########################################################################################################################
# Trajectory time
T = 80

# Bool: unsupervised or not
unsupervised = True

# Noise Ratio q^2/r^2 [dB]
ratio  = 0

# Range of desired 1/r^2 [dB]
R_decibel_train_ranges = np.array([-10,-3,0,3,10,20,30])

# File specification
data_file_specification = 'Ratio_{}---R_{}---T_{}'
model_file_specification = 'Ratio_{}---R_{}---T_{}---unsupervised_{}'

# Training device
hardware = 'cuda:0'
# hardware = 'cpu'
device = torch.device(hardware)

########################################################################################################################
### For online use case
########################################################################################################################
T_online = 10_000
N_E_online = 100
N_T_online = 100

# parameters of the pretrained model
rdB_pretrained = -10
ratio_pretrained = 0

# data parameter
rdB_data = -25
ratio_data = -15

# Time step on which to start the training
training_start = 4000


########################################################################################################################

# Dataset Class for easy batch loading and randomization
class Dataset(torch.utils.data.Dataset):
    def __init__(self,input,target):
        # device = device
        self.input = input.to(device)
        self.target = target.to(device)

    def __getitem__(self, item):
        return self.input[item],self.target[item]

    def __len__(self):
        return self.input.size()[0] if self.input.size()[0] == self.target.size()[0] else None


########################################################################################################################
def DataGen(SysModel_data, fileName,N_E = N_E, N_CV = N_CV, N_T = N_T):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E,'Training')
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV,'Cross Val')
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T,'Testing')
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    #################
    ### Save Data ###
    #################
    torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.load(fileName)
    train_data = Dataset(training_input,training_target)
    cv_data =  Dataset(cv_input,cv_target)
    test_data = Dataset(test_input,test_target)

    return train_data,cv_data,test_data