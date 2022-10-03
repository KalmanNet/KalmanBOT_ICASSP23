import torch
import torch.nn as nn
import numpy as np
import random
from Plot import Plot
import time
from torch.utils.tensorboard import SummaryWriter
import sys
from position_MLP import position_MLP
import sys


if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("using GPU!")
else:
    dev = torch.device("cpu")
    print("using CPU!")


class Pipeline_kf_train:

    def __init__(self, folderName, modelName, positionName):
        super().__init__()
        # self.runtime_index = index
        # self.Time = Time
        self.folderName = folderName
        self.modelName = modelName
        self.positionName = positionName
        self.modelFileName = self.folderName + self.modelName + ".pt"
        self.positionModelName = self.folderName + self.positionName + ".pt"
        # self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def setssModel(self, ssModel):
        self.ssModel = ssModel
    def setMoment(self, beta_0, R_0):
        self.m1x_0 = beta_0
        self.m2x_0 = R_0

    def setModel(self, model):
        self.model = model.to(dev, non_blocking=True)

    def set_positionModel(self, position_model):
        self.position_model = position_model.to(dev, non_blocking=True)

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay, choice = 0):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay, capturable=True)
        if choice==1: # only train position model
            self.optimizer = torch.optim.Adam(self.position_model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay, capturable=True)
 

    
    def pnl(self, positions, train_yx, beta):
        positions = positions[0,0,:]#.clone()
            
        # pnl = torch.zeros_like(positions)
        tmp = torch.transpose(-beta[0,0:1,:], 1,0)
        temp = torch.hstack([tmp, torch.ones(tmp.shape[0],1)])
        position = temp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
        asset_price = torch.transpose(train_yx[0], 1, 0)[:,:-1][:, [1,0]]
        # asset_price = np.sum(asset_price, axis = 1)
        asset_price_diff = torch.diff(asset_price.T).T
        # pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)

        # 计算收益的数值
        res = torch.sum(torch.sum(asset_price_diff * position[:-1], axis = 1))
        # cum_pnl = pnl
        # cum_pnl[0] = 0
        # cum_pnl = torch.cumsum(cum_pnl, dim=0)
        # print('cum_pnl:', res.item())
        # res.backward()
        return res

    
    def NNTrain(self, n_Examples, training_dataset, n_CV, cv_dataset):

        self.N_E = n_Examples
        self.MSE_train_linear_epoch_obs = np.empty([self.N_Epochs])
        self.MSE_train_dB_epoch_obs = np.empty([self.N_Epochs])

        # Setup Dataloader
        train_data = torch.utils.data.DataLoader(training_dataset,batch_size = self.N_B, shuffle = False, generator=torch.Generator(device='cuda')) #, generator=torch.Generator(device='cuda')
       
        ##############
        ### Epochs ###
        ##############
        self.MSE_train_opt = 1000
        self.MSE_train_idx_opt = 0
        # torch.autograd.set_detect_anomaly(True)
        for ti in range(0, self.N_Epochs):

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            self.model.InitSequence(self.m1x_0.clone(), self.m2x_0.clone())

            # Init Hidden State
            # self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            # Load random batch sized data, creating new iter ensures the data is shuffled
            train_yx = next(iter(train_data))
            y_training = train_yx[:,0:1,:]

            positions = torch.zeros(1, 1, self.model.T, device=self.model.device)
            x_out_training = torch.empty(self.N_B, 2, self.model.T, device=self.model.device)
            
            for t in range(0, self.model.T):
                dy, x_out, S = self.model(train_yx[:,:,t])
                if t>0:
                    positions[:,:,t] = self.position_model(dy, S, positions[:,:,t-1].clone())
                x_out_training[:,:,t] = x_out.T

            # Compute Training Loss
            # LOSS = positions[0,0,-1]
            LOSS = -self.pnl(positions, train_yx, x_out_training)
            # print(self.model.q.item(), self.model.r.item())

            # Average
            
            ##################
            ### Optimizing ###
            ##################
            self.optimizer.zero_grad()
            LOSS.backward() #retain_graph=True      
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
          
            # train_print = self.MSE_train_linear_epoch_obs[ti]
            print(ti, "PnL Training :", -LOSS.item())

            # reset hidden state gradient
            # self.model.hn.detach_()
            self.model.elements_detach()

            torch.save(self.position_model, self.positionModelName)

            # Reset the optimizer for faster convergence
            if ti % 50 == 0 and ti != 0:
                self.ResetOptimizer()
                print('Optimizer has been reset')
