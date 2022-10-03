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


class Pipeline_trading:

    def __init__(self, folderName, modelName, positionName):
        super().__init__()
        # self.runtime_index = index
        # self.Time = Time
        self.folderName = folderName
        self.modelName = modelName
        self.positionName = positionName
        self.modelFileName = self.folderName + self.modelName + ".pt"
        self.positionModelName = self.folderName + self.positionName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def save_modified(self,i,index):
        torch.save(self, self.folderName + "pipeline_q"+str(i)+'_r'+str(index) + ".pt")

    def set_index(self, i, index):
        self.i = i
        self.index = index

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model.to(dev, non_blocking=True)

    def set_positionModel(self, position_model):
        self.position_model = position_model.to(dev, non_blocking=True)

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay, learnable_pos=0):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay
        self.learnable_pos = learnable_pos
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
    
        self.optimizer = torch.optim.Adam(list(self.model.parameters())+list(self.position_model.parameters()), lr=self.learningRate, weight_decay=self.weightDecay, capturable=True)
    
    def pnl(self, positions, train_yx, beta):
        positions = positions[0,0,:]

        pnl = torch.zeros_like(positions)
        tmp = torch.transpose(-beta[0,0:1,:], 1,0)
        tmp = torch.hstack([tmp, torch.ones(tmp.shape[0],1)])
        position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
        asset_price = torch.transpose(train_yx[0], 1, 0)[:,:-1][:, [1,0]]
        # asset_price = np.sum(asset_price, axis = 1)
        asset_price_diff = torch.diff(asset_price.T).T
        pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)

        # 计算收益的数值
        cum_pnl = pnl
        cum_pnl[0] = 0
        cum_pnl = torch.cumsum(cum_pnl, dim=0)
        # print('cum_pnl:', cum_pnl[-1])
        return cum_pnl[-1]

    def pnl_limit(self, positions, train_yx, beta):
        positions = positions[0,0,:]

        pnl = torch.zeros_like(positions)
        tmp = torch.transpose(-beta[0,0:1,:], 1,0)
        tmp = torch.hstack([-torch.ones(tmp.shape[0],1), torch.ones(tmp.shape[0],1)])
        position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
        asset_price = torch.transpose(train_yx[0], 1, 0)[:,:-1][:, [1,0]]
        # asset_price = np.sum(asset_price, axis = 1)
        asset_price_diff = torch.diff(asset_price.T).T
        pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)

        # 计算收益的数值
        cum_pnl = pnl
        cum_pnl[0] = 0
        cum_pnl = torch.cumsum(cum_pnl, dim=0)
        # print('cum_pnl:', cum_pnl[-1])
        return cum_pnl[-1]

    
    def NNTrain(self, n_Examples, training_dataset, n_CV, cv_dataset):

        self.N_E = n_Examples
        self.MSE_train_linear_epoch_obs = np.empty([self.N_Epochs])
        self.MSE_train_dB_epoch_obs = np.empty([self.N_Epochs])

        # Setup Dataloader
        train_data = torch.utils.data.DataLoader(training_dataset, batch_size = self.N_B, shuffle = False, generator=torch.Generator(device='cuda')) #, generator=torch.Generator(device='cuda')
       
        ##############
        ### Epochs ###
        ##############
        self.MSE_train_opt = 1000
        self.MSE_train_idx_opt = 0

        for ti in range(0, self.N_Epochs):

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            # Load random batch sized data, creating new iter ensures the data is shuffled
            train_yx = next(iter(train_data))
            y_training = train_yx[:,0:1,:]
            # self.model.SetBatch(self.N_B)
            self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T)

            positions = torch.zeros(1, 1, self.ssModel.T, device=self.model.device)
            x_out_training = torch.empty(self.N_B,self.ssModel.m, self.ssModel.T, device=self.model.device)
            
            for t in range(0, self.ssModel.T):
                # print(t)
                dy, x_out, S = self.model(train_yx[:,:,t])
                # print(dy, x_out, S)
                if self.learnable_pos==0:
                    dy = dy * 100
                if t>0:
                    positions[:,:,t] = self.position_model(dy, S, positions[:,:,t-1].clone())
                x_out_training[:,:,t] = x_out.T
                # positions[:,:,t] = self.position_model(dy)
                # x_out_training[:,:,t] = x_out.T
                # y_out_training[:,:,t] = self.model.m1y.squeeze().T

            # LOSS = -self.pnl(positions, train_yx, x_out_training)
            LOSS = -self.pnl(positions, train_yx, x_out_training)
            self.MSE_train_linear_epoch_obs[ti] = LOSS
            self.optimizer.zero_grad()
            LOSS.backward()
            self.optimizer.step()
            train_print = self.MSE_train_linear_epoch_obs[ti]
            print(ti, "PnL Training :", -train_print)

            # reset hidden state gradient
            # self.model.hn.detach_()

            # Reset the optimizer for faster convergence
            if ti % 50 == 0 and ti != 0:
                self.ResetOptimizer()
                print('Optimizer has been reset')

            torch.save(self.model, self.modelFileName)
            torch.save(self.position_model, self.positionModelName)

    def NNTest(self, test_input, test_target):
        test_target = test_target.to(dev, non_blocking=True)
        self.N_T = test_input.size()[0]

        self.MSE_test_linear_arr = torch.empty([self.N_T])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')
             
        self.model = torch.load(self.modelFileName, map_location=dev)
        
        self.model.eval()

        torch.no_grad()
        
        x_out_array = torch.empty(self.N_T,self.ssModel.m, self.ssModel.T_test)
        start = time.time()
        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j, :, :]

            self.model.InitSequence(self.ssModel.m1x_0, self.ssModel.T_test)

            x_out_test = torch.empty(self.ssModel.m, self.ssModel.T_test).to(dev, non_blocking=True)

            for t in range(0, self.ssModel.T_test):
                x_out_test[:, t] = self.model(y_mdl_tst[:, t])

            self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()
            x_out_array[j,:,:] = x_out_test
        
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
        self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_linear_avg+self.MSE_test_linear_std)-self.MSE_test_dB_avg

        # Print MSE Cross Validation
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        # Print std
        str = self.modelName + "- STD Test:" 
        print(str, self.MSE_test_dB_std, "[dB]")
        # Print Run Time
        print("Inference Time:", t)
        

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_array]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)