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

class position_MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device
        self.to(self.device)
        self.model1 = nn.Sequential(
            Linear(1,16),
            nn.ReLU(),
            Linear(16,1)
        )
        
    def forward(self, dy):
        return self.model1(dy)




