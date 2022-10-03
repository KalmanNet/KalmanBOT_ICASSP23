import torch
import torch.nn as nn
import numpy as np
from Plot import Plot


class Pipeline_Offline:

    def __init__(self, folderName, modelName):
        super().__init__()
        # self.Time = Time
        self.folderName = folderName + '\\'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)
        torch.save(self.model,self.modelFileName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel


    def setModel(self, model):
        self.model = model
        self.model.to(self.model.device)

    def ResetOptimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay,unsupervised):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        # self.loss_fn = self.max_morm()

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.unsupervised = unsupervised

    def NNTrain(self, n_Examples, training_dataset,n_CV,cv_dataset):

        self.N_E = n_Examples
        if not self.unsupervised:
            self.N_CV = n_CV


            self.MSE_cv_linear_epoch = np.empty([self.N_Epochs])
            self.MSE_cv_dB_epoch = np.empty([self.N_Epochs])

            self.MSE_cv_linear_epoch_obs = np.empty([self.N_Epochs])
            self.MSE_cv_dB_epoch_obs = np.empty([self.N_Epochs])

            self.MSE_train_linear_epoch = np.empty([self.N_Epochs])
            self.MSE_train_dB_epoch = np.empty([self.N_Epochs])

        

        self.MSE_train_linear_epoch_obs = np.empty([self.N_Epochs])
        self.MSE_train_dB_epoch_obs = np.empty([self.N_Epochs])

        # Setup Dataloader
        train_data = torch.utils.data.DataLoader(training_dataset,batch_size = self.N_B,shuffle = True, generator=torch.Generator(device='cuda'))
        if not self.unsupervised:
            cv_data = torch.utils.data.DataLoader(cv_dataset,batch_size = self.N_CV,shuffle = False)

        ##############
        ### Epochs ###
        ##############
        if not self.unsupervised:
            self.MSE_cv_dB_opt = 1000
            self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):

            if not self.unsupervised:
            #################################
            ### Validation Sequence Batch ###
            #################################

                # Cross Validation Mode
                self.model.eval()


                # Load obserations and targets from CV data
                y_cv,cv_target = next(iter(cv_data))
                self.model.SetBatch(self.N_CV)
                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_cv = torch.empty(self.N_CV,self.ssModel.m, self.ssModel.T,device= self.model.device)
                y_out_cv = torch.empty(self.N_CV,self.ssModel.n, self.ssModel.T,device= self.model.device)

                for t in range(0, self.ssModel.T):
                    x_out_cv[:,:, t] = self.model(y_cv[:,:, t]).T
                    y_out_cv[:,:,t] = self.model.m1y.squeeze().T

                # Compute Training Loss
                cv_loss = self.loss_fn(x_out_cv[:,:,:self.ssModel.T], cv_target[:,:,:self.ssModel.T]).item()
                cv_loss_obs =  self.loss_fn(y_out_cv[:,:,:self.ssModel.T], y_cv[:,:,:self.ssModel.T]).item()

                # Average
                self.MSE_cv_linear_epoch[ti] = np.mean(cv_loss)
                self.MSE_cv_dB_epoch[ti] = 10 * np.log10(self.MSE_cv_linear_epoch[ti])

                self.MSE_cv_linear_epoch_obs[ti] = np.mean(cv_loss_obs)
                self.MSE_cv_dB_epoch_obs[ti] = 10*np.log10(self.MSE_cv_linear_epoch_obs[ti])

                relevant_loss = cv_loss_obs if self.unsupervised else cv_loss
                relevant_loss = 10 * np.log10(relevant_loss)

                if (relevant_loss < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = relevant_loss
                    self.MSE_cv_idx_opt = ti
                    torch.save(self.model, self.modelFileName)



            ###############################
            ### Training Sequence Batch ###
            ###############################

            torch.save(self.model, self.modelFileName)

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            # Load random batch sized data, creating new iter ensures the data is shuffled
            if not self.unsupervised:
                y_training, train_target = next(iter(train_data)) ####ORIGINAL code
            else:
                train_yx = next(iter(train_data))
                y_training = train_yx[:,0:1,:]
            self.model.SetBatch(self.N_B)
            self.model.InitSequence(self.ssModel.m1x_0)

            x_out_training = torch.empty(self.N_B,self.ssModel.m, self.ssModel.T,device= self.model.device)
            y_out_training = torch.empty(self.N_B,self.ssModel.n, self.ssModel.T,device=self.model.device)

            for t in range(0, self.ssModel.T):
                x_out_training[:,:, t] = self.model(train_yx[:,:, t]).T
                y_out_training[:,:,t] = self.model.m1y.squeeze().T


            # Compute Training Loss
            if not self.unsupervised:
                loss  = self.loss_fn(x_out_training[:,:,:self.ssModel.T], train_target[:,:,:self.ssModel.T])
            loss_obs  = self.loss_fn(y_out_training[:,:,:self.ssModel.T], y_training[:,:,:self.ssModel.T])

            # Select loss, from which to update the gradient
            if not self.unsupervised:
                LOSS = loss_obs if self.unsupervised else loss
            else:
                LOSS = loss_obs

            # Average
            if not self.unsupervised:
                self.MSE_train_linear_epoch[ti] = loss
                self.MSE_train_dB_epoch[ti] = 10 * np.log10(self.MSE_train_linear_epoch[ti])

            self.MSE_train_linear_epoch_obs[ti] = loss_obs
            self.MSE_train_dB_epoch_obs[ti] = 10*np.log10(self.MSE_train_linear_epoch_obs[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            # Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            LOSS.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            if not self.unsupervised:
                train_print = self.MSE_train_dB_epoch_obs[ti] if self.unsupervised else self.MSE_train_dB_epoch[ti]
                cv_print = self.MSE_cv_dB_epoch_obs[ti] if self.unsupervised else self.MSE_cv_dB_epoch[ti]
                print(ti, "MSE Training :", train_print, "[dB]", "MSE Validation :", cv_print,"[dB]")
            else:
                train_print = self.MSE_train_dB_epoch_obs[ti]
                print(ti, "MSE Training :", train_print, "[dB]")

            if (ti > 1):
                if not self.unsupervised:
                    d_train = self.MSE_train_dB_epoch_obs[ti] - self.MSE_train_dB_epoch_obs[ti - 1] if self.unsupervised \
                            else self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]


                    d_cv = self.MSE_cv_dB_epoch_obs[ti] - self.MSE_cv_dB_epoch_obs[ti - 1] if self.unsupervised \
                            else self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]

                    print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")
                else:
                    d_train = self.MSE_train_dB_epoch_obs[ti] - self.MSE_train_dB_epoch_obs[ti - 1]
                    print("diff MSE Training :", d_train, "[dB]")

            if not self.unsupervised:
                print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

            # reset hidden state gradient
            self.model.hn.detach_()

            # Reset the optimizer for faster convergence
            if ti % 50 == 0 and ti != 0:
                self.ResetOptimizer()
                print('Optimizer has been reset')

    def NNTest(self, n_Test, test_dataset):
        with torch.no_grad():

            self.N_T = n_Test

            # Load test data and create iterator
            test_data = torch.utils.data.DataLoader(test_dataset,batch_size = self.N_T,shuffle = False)
            test_data_iter = iter(test_data)

            # Allocate Array
            self.MSE_test_linear_arr = torch.empty([self.N_T],device = self.model.device)
            self.MSE_test_linear_arr_obs = torch.empty([self.N_T],device= self.model.device)

            # MSE LOSS Function
            loss_fn = nn.MSELoss(reduction='none')

            self.model.eval()

            # Load training data from iter
            test_input,test_target = next(test_data_iter)
            test_target = test_target.to(self.model.device)
            test_input = test_input.to(self.model.device)

            self.model.SetBatch(self.N_T)
            self.model.InitSequence(self.ssModel.m1x_0)

            x_out_test = torch.empty(self.N_T,self.ssModel.m, self.ssModel.T,device=self.model.device)
            y_out_test = torch.empty(self.N_T,self.ssModel.n, self.ssModel.T,device=self.model.device)

            for t in range(0, self.ssModel.T):
                x_out_test[:,:, t] = self.model(test_input[:,:, t]).T
                y_out_test[:,:,t] = self.model.m1y.T

            loss_unreduced = loss_fn(x_out_test[:,:,:self.ssModel.T],test_target[:,:,:self.ssModel.T])
            loss_unreduced_obs = loss_fn(y_out_test[:,:,:self.ssModel.T],test_input[:,:,:self.ssModel.T])

            # Create the linear loss from the total loss for the batch
            loss = torch.mean(loss_unreduced,axis = (1,2))
            loss_obs = torch.mean(loss_unreduced_obs,axis = (1,2))


            self.MSE_test_linear_arr[:] = loss
            self.MSE_test_linear_arr_obs[:] = loss_obs

            # Average
            self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
            self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg).item()

            self.MSE_test_linear_avg_obs = torch.mean(self.MSE_test_linear_arr_obs)
            self.MSE_test_dB_avg_obs = 10 * torch.log10(self.MSE_test_linear_avg_obs).item()

            # Print MSE Cross Validation
            str = self.modelName + "-" + "MSE Test:"
            print(str, self.MSE_test_dB_avg, "[dB]")


    def PlotTrain(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)
