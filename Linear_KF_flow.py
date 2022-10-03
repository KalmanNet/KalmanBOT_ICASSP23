import torch
import torch.nn as nn
import torch.nn.functional as func
import gc
from KalmanNet_data import device

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

class KF_flow(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.device = device
        self.to(self.device)

    def init_SS(self, F, H, T, ratio = 1):
        self.F = F;
        self.F_T = torch.transpose(self.F, 0, 1);#!!!!!!!!!!!
        # self.F_T = self.F
        self.m = 2

        self.q = nn.Parameter(torch.tensor([4e-4]), requires_grad=True)
        # self.q = torch.tensor([1e-3]) 
        # self.q.requires_grad = True
        self.Q = self.q * self.q * torch.eye(self.m)

        self.H = H;
        self.H_T = torch.transpose(self.H, 0, 1);

        self.n = 1

        self.r = nn.Parameter(torch.tensor([1e-3]), requires_grad=True)
        # self.r = torch.tensor([1e-3])
        # self.r.requires_grad = True
        self.R = self.r * self.r * torch.eye(self.n)

        self.T = T
        self.T_test = self.T

        self.ratio = ratio # every self.ratio times of prediction and one innovation
        
        # self.innovations = torch.empty(size=[self.n, self.T_test]).to(dev)
        # self.y_vars = torch.empty(size=[self.n, self.n, self.T_test]).to(dev)
    def set_qr(self, q, r):
        self.q = nn.Parameter(torch.tensor([q]), requires_grad=True)
        self.Q = self.q * self.q * torch.eye(self.m)
        self.r = nn.Parameter(torch.tensor([r]), requires_grad=True)
        self.R = self.r * self.r * torch.eye(self.n)

    # Predict

    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior);

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior);
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q;

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior);

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior);
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R;

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
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

    def Update(self, y):
        self.H = y[:,1:]
        y = y[:,0:1]
        self.H_T = torch.transpose(self.H, 1, 0)
        self.Predict();
        self.KGain();
        self.Innovation(y);
        self.Correct();

        return self.m1x_posterior,self.m2x_posterior;

    def forward(self, y):
        y = y.to(dev, non_blocking=True)
        self.Update(y)

        # print(self.dy)
        # self.dy.backward()
        # self.m1x_posterior.sum().backward()
        # self.m2y.backward()

        return self.dy, self.m1x_posterior, self.m2y

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

    def elements_detach(self):
        self.m1x_prior.detach_()
        self.m2x_prior.detach_()
        self.m1y.detach_()
        self.m2y.detach_()
        self.m2x_posterior.detach_()
        self.KG.detach_()
        self.dy.detach_()
        self.m1x_posterior.detach_()
        self.Q.detach_()
        self.Q = self.q * self.q * torch.eye(self.m)
        self.R.detach_()
        self.R = self.r * self.r * torch.eye(self.n)

    #########################
    ### Generate Sequence ###
    #########################
    # def GenerateSequence(self, y, T):
    #     # Pre allocate an array for predicted state and variance
    #     self.x = torch.empty(size=[self.m, T]).to(dev)
    #     self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

    #     self.innovations = torch.empty(T).to(dev)
    #     self.y_vars = torch.empty(T).to(dev)

    #     self.m1x_posterior = self.m1x_0
    #     self.m2x_posterior = self.m2x_0

        # for t in range(0, T):
        #     yt = torch.unsqueeze(y[:, t], 1);  # yt = y[t]#####!!!!!!!!!!!!!!!
        #     H = torch.tensor(self.H[t]).unsqueeze(0)
        #     H_T = torch.transpose(H, 0, 1)

        #     xt,sigmat = self.Update(yt, H, H_T);
        #     self.x[:, t] = torch.squeeze(xt)
        #     self.sigma[:, :, t] = torch.squeeze(sigmat)
        #     self.innovations[t] = torch.squeeze(self.dy)
        #     self.y_vars[t] = torch.squeeze(self.m2y)