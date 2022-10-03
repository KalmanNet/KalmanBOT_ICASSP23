import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules.linear import Linear
from KalmanNet_data import device

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

# in_mult = 5
# out_mult = 40

class position_learn_bollinger(torch.nn.Module):
    def __init__(self, std, scale):
        super().__init__()
        self.device = device
        self.to(self.device)
        self.std = std
        self.scale = nn.Parameter(torch.tensor([scale]), requires_grad=True)
  
    def indicator_0(self, last_pos):
        return torch.exp(-1/(self.std**2*2)*((last_pos)**2)) # 1/torch.sqrt(2*torch.pi)/self.std*
        # if last_pos==torch.tensor([0.]): return 1
        # else: return 0

    def indicator_1(self, last_pos):
        return torch.exp(-1/(self.std**2*2)*((last_pos-1)**2))
        # if last_pos==torch.tensor([1.]): return 1
        # else: return 0
        
    def indicator__1(self, last_pos):
        return torch.exp(-1/(self.std**2*2)*((last_pos+1)**2))
        # if last_pos==torch.tensor([-1.]): return 1
        # else: return 0

    def u(self, z):
        return torch.distributions.normal.Normal(0, 0.01*self.std, validate_args=None).cdf(z)
        # return 0.5*torch.tanh(z)+0.5
        # return torch.heaviside(z, torch.tensor(1.))

    def forward(self, delta, s, last_pos):
        z = self.scale * delta
        a = (-self.u(z-torch.sqrt(s))+self.u(-z))*self.indicator_1(last_pos)
        b = (-self.u(z-torch.sqrt(s))+self.u(-z-torch.sqrt(s)))*self.indicator_0(last_pos)
        c = (-self.u(z)+self.u(-z-torch.sqrt(s)))*self.indicator__1(last_pos)
        return a+b+c




