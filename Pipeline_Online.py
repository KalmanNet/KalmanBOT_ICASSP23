import torch
import torch.nn as nn
import copy
import numpy as np
import random
from Plot import Plot
from tqdm import trange

class Pipeline_Online:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
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

    def setTrainingParams(self, learningRate, weightDecay, stride,training_start = 0):

        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        # self.loss_fn = self.max_morm()

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.stride = stride
        self.training_start = training_start

    def NNTrain(self, n_Examples, training_dataset):

        # Load each trajectory one-by-one
        train_data = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=False, pin_memory=True)
        train_data_iter = iter(train_data)

        # Allocate estimate array
        self.output_predictions = torch.empty((n_Examples, self.ssModel.n, self.ssModel.T), requires_grad=False)
        self.state_predictions = torch.empty((n_Examples, self.ssModel.m, self.ssModel.T), requires_grad=False)

        # Copy to restore the NN to its original state for each trajectory
        original_model = copy.deepcopy(self.model)

        # For printing out useful information
        counter = 0

        # Start looping over trajectories
        for trajectorie in range(n_Examples):
            print('Trajectory: ', trajectorie + 1, '/', n_Examples)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Reset the model
            self.model = copy.deepcopy(original_model)

            # Reset optimizer
            self.ResetOptimizer()

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            # Load the next trajectory
            y_training, train_target = next(train_data_iter)

            # Set Batch size to 1 (single trajectory)
            self.model.SetBatch(1)

            # Initialize state
            self.model.InitSequence(self.ssModel.m1x_0)

            # Calculate the number of strides required
            number_of_stride = int(self.ssModel.T / self.stride)

            # Calculate the remainder
            remainder = int(self.ssModel.T % self.stride)

            # Go through the whole trajectory stride by stride, updating the NN parameters after every stride-
            # time steps
            for stride in range(number_of_stride):

                # Initialize training mode
                self.model.train()

                # Set the initial posterior to the previous posterior and detaching it from the gradient calculation
                self.model.InitSequence(self.model.m1x_posterior.detach())

                # Get next observations
                observations = y_training[0, :,(stride * self.stride):(stride * self.stride + self.stride)]
                observations = observations.reshape(1,self.ssModel.n,self.stride).detach()

                # Initialize hidden state of GRU
                self.model.init_hidden()

                # Allocate estimate arrays
                x_out_online = torch.empty(1, self.ssModel.m, self.stride)
                y_out_online = torch.zeros(1, self.ssModel.n, self.stride)

                # Loop trough a single stride
                for t in range(self.stride):

                    # Take time step in NN
                    x_out_online[:, :, t] = self.model(observations[:, :, t]).T

                    # Get the output estimate from the NN
                    y_out_online[:, :, t] = self.model.m1y.squeeze().T

                # Plug obtained values into the allocated arrays
                self.output_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = y_out_online.detach()

                self.state_predictions[trajectorie, :,
                (stride * self.stride):(stride * self.stride + self.stride)] = x_out_online.detach()

                # Calculate Loss
                LOSS = self.loss_fn(y_out_online, observations)

                # Print statistics every 10% of a trajectory
                counter += 1
                if counter % max(int(number_of_stride/10),1) == 0:
                    print('Training itt:', stride + 1, '/', number_of_stride, ',OBS MSE:',
                          10 * torch.log10(LOSS).item(), '[dB]')

                # optimize if t > training start
                if stride*self.stride >= self.training_start:
                    # Zero Gradient
                    self.optimizer.zero_grad()
                    # optimize
                    LOSS.backward()
                    self.optimizer.step()

                # Clear variables to save memory
                del observations, y_out_online, LOSS, x_out_online
            # Calculate the final time steps
            if not remainder == 0:

                # Initialize the posterior
                self.model.InitSequence(self.model.m1x_posterior.detach())

                # Get Observations
                observations = y_training[0, :, -remainder:].reshape(1, self.ssModel.n, remainder).detach()

                # Initialize hidden state of GRU
                self.model.init_hidden()

                # Allocate estimates
                x_out_online = torch.empty(1, self.ssModel.m, remainder)
                y_out_online = torch.empty(1, self.ssModel.n, remainder)

                # Loop through the remaining time steps
                for t in range(remainder):
                    # Take time step in NN
                    x_out_online[:, :, t] = self.model(observations[:, :, t]).T
                    # Get the output of the NN
                    y_out_online[:, :, t] = self.model.m1y.squeeze().T

                # Plug obtained values into the allocated arrays
                self.output_predictions[trajectorie, :, -remainder:] = y_out_online
                self.state_predictions[trajectorie, :, -remainder:] = x_out_online

            # Reset the optimizer for the next trajectory
            self.ResetOptimizer()

        loss_fn = torch.nn.MSELoss(reduction='none')

        self.MSE_state_arr = loss_fn(training_dataset.target,self.state_predictions)
        self.MSE_observation_arr = loss_fn(training_dataset.input,self.output_predictions)

        self.MSE_states_over_time = 10 * torch.log10(torch.mean(self.MSE_state_arr,axis = (0,1)))
        self.MSE_observation_over_time = 10 * torch.log10(torch.mean(self.MSE_observation_arr,axis = (0,1)))

        self.MSE_states_over_trajectories = 10 * torch.log10(torch.mean(self.MSE_state_arr,axis = (1,2)))
        self.MSE_observation_over_trajectories = 10 * torch.log10(torch.mean(self.MSE_observation_arr,axis = (1,2)))


        self.MSE_states_before_training = 10 * torch.log10(torch.mean(self.MSE_state_arr[:,:,:self.training_start])).item()
        self.MSE_states_after_training = 10 * torch.log10(torch.mean(self.MSE_state_arr[:,:,self.training_start:])).item()

        if not self.training_start==0:
            print('MSE before training start:',self.MSE_states_before_training,'[dB]')
        print('MSE after training start:', self.MSE_states_after_training,'[dB]')


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





