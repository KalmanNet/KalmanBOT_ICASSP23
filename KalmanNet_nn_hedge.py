import torch
import torch.nn as nn
import torch.nn.functional as func
import gc
from KalmanNet_data import device


class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = device
        self.to(self.device)
    #############
    ### Build ###
    #############
    def Build(self, ssModel):



        self.InitSystemDynamics(ssModel.F, ssModel.H)

        self.ssModel = ssModel

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (10)

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True).to(self.device,non_blocking = True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers,batch_first= True).to(self.device,non_blocking = True)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True).to(self.device,non_blocking = True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True).to(self.device,non_blocking = True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device,non_blocking = True)
        self.H_T = torch.transpose(H, 0, 1)
        self.n = 1

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):

        # Adjust for batch size
        M1_0 = torch.cat(self.batch_size*[M1_0],axis = 1).reshape(self.m,self.batch_size)

        self.m1x_prior = M1_0.detach().to(self.device,non_blocking = True)

        self.m1x_posterior = M1_0.detach().to(self.device,non_blocking = True)

        self.state_process_posterior_0 = M1_0.detach().to(self.device,non_blocking = True)

    #########################################################
    ### Set Batch Size and initialize hidden state of GRU ###
    #########################################################

    def SetBatch(self,batch_size):

        self.batch_size = batch_size

        self.hn = torch.randn(self.seq_len_hidden,self.batch_size,self.hidden_dim,requires_grad=False).to(self.device)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(self.F,self.state_process_posterior_0)

        # Compute the 1-st moment of y based on model knowledge and without noise
        self.obs_process_0 = torch.matmul(self.H, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior)


    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Normalize y
        dm1y = y.squeeze() - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)


        # print(dm1y_norm.unsqueeze(0))
        # print(dm1x_norm)
        # KGain Net Input
        if self.batch_size==1:
            KGainNet_in = torch.cat([dm1y_norm.unsqueeze(0), dm1x_norm], dim=0)
        else:
            KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in.T)


        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size,self.m, self.n))
        del KG,KGainNet_in,dm1y,dm1x,dm1y_norm,dm1x_norm,dm1x_reshape


    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):

        self.H = y[:,1:]
        
        # print(self.H)
        self.H_T = torch.transpose(self.H, 0, 1)
        # print(self.H_T)
        y = y[:,0:1]

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y
        self.innovations = dy

        # Compute the 1-st posterior moment
        # Initialize array of Innovations
        INOV = torch.empty((self.m,self.batch_size),device= self.device)

        for batch in range(self.batch_size):
            # Calculate the Inovation for each KGain
            INOV[:,batch] = torch.matmul(self.KGain[batch],dy[:,batch]).squeeze()

        self.m1x_posterior = self.m1x_prior + INOV

        del INOV,dy,y

        return torch.squeeze(self.m1x_posterior)


    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in);
        La1_out = self.KG_relu1(L1_out);

        ###########
        ### GRU ###
        ###########
        GRU_in = La1_out.reshape((self.batch_size,self.seq_len_input,self.input_dim))
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (self.batch_size, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        self.L3_out = self.KG_l3(La2_out)
        del L2_out,La2_out,GRU_out,GRU_in,GRU_out_reshape,L1_out,La1_out
        return self.L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.to(self.device,non_blocking = True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
