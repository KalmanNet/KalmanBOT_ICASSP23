"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

class KalmanFilter:

    def __init__(self, SystemModel, ratio = 1):
        self.F = SystemModel.F;
        self.F_T = torch.transpose(self.F, 0, 1);#!!!!!!!!!!!
        # self.F_T = self.F
        self.m = SystemModel.m

        self.Q = SystemModel.Q;

        self.H = SystemModel.H;
        # self.H_T = torch.transpose(self.H, 0, 1);

        self.n = SystemModel.n

        self.R = SystemModel.R;

        self.T = SystemModel.T;
        self.T_test = SystemModel.T_test;

        self.ratio = ratio # every self.ratio times of prediction and one innovation
        
        # self.innovations = torch.empty(size=[self.n, self.T_test]).to(dev)
        # self.y_vars = torch.empty(size=[self.n, self.n, self.T_test]).to(dev)

    # Predict

    def Predict(self, H, H_T):
        m1x_temp = self.m1x_posterior
        m2x_temp = self.m2x_posterior
        for i in range(self.ratio):
            if i==self.ratio-1:
                # Predict the 1-st moment of x
                self.m1x_prior = torch.matmul(self.F, m1x_temp);

                # Predict the 2-nd moment of x
                self.m2x_prior = torch.matmul(self.F, m2x_temp);
                self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q;

                # Predict the 1-st moment of y
                self.m1y = torch.matmul(H, self.m1x_prior);

                # Predict the 2-nd moment of y
                self.m2y = torch.matmul(H, self.m2x_prior);
                # print(self.R)
                self.m2y = torch.matmul(self.m2y, H_T) + self.R;
                
            else:
                # Predict the 1-st moment of x
                m1x_temp = torch.matmul(self.F, m1x_temp);

                # Predict the 2-nd moment of x
                m2x_temp = torch.matmul(self.F, m2x_temp);
                m2x_temp = torch.matmul(m2x_temp, self.F_T) + self.Q;

        # for i in range(self.ratio):
        #     # Predict the 1-st moment of x
        #     self.m1x_prior = torch.matmul(self.F, self.m1x_posterior);

        #     # Predict the 2-nd moment of x
        #     self.m2x_prior = torch.matmul(self.F, self.m2x_posterior);
        #     self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q;

        #     # Predict the 1-st moment of y
        #     self.m1y = torch.matmul(self.H, self.m1x_prior);

        #     # Predict the 2-nd moment of y
        #     self.m2y = torch.matmul(self.H, self.m2x_prior);
        #     self.m2y = torch.matmul(self.m2y, self.H_T) + self.R;

    # Compute the Kalman Gain
    def KGain(self, H_T):
        self.KG = torch.matmul(self.m2x_prior, H_T)
        # print(self.KG)
        # print(self.m2y)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y;
        # self.innovations[:]

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy);

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y, H, H_T):
        self.Predict(H,H_T);
        self.KGain(H_T);
        self.Innovation(y);
        self.Correct();

        return self.m1x_posterior,self.m2x_posterior;

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        self.innovations = torch.empty(T).to(dev)
        self.y_vars = torch.empty(T).to(dev)

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(0, T):
            yt = torch.unsqueeze(y[:, t], 1);  # yt = y[t]#####!!!!!!!!!!!!!!!
            H = torch.tensor(self.H[t]).unsqueeze(0)
            H_T = torch.transpose(H, 0, 1)

            xt,sigmat = self.Update(yt, H, H_T);
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)
            self.innovations[t] = torch.squeeze(self.dy)
            self.y_vars[t] = torch.squeeze(self.m2y)