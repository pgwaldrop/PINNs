import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#form torchvision import datasets, transform
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt

#***********************************************************
# Plotting a PINN trained to solve the pendulum equation (P. Waldrop)
# L * d^2u/dt^2 + g * sin(u) = 0
#***********************************************************


# Neural network
class Net(nn.Module):
    def __init__(self, D=2, W=2):
        super(Net, self).__init__()

        layers = []

        layers.append(nn.Linear(1,W))
        layers.append(nn.Sigmoid())

        for i in range(D-2):
            layers.append(nn.Linear(W,W))
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(W,1))
        #layers.append(nn.Softplus())

        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)
    
model = Net(4,256)
model.load_state_dict(torch.load('pendulum.pt'))

ts = torch.linspace(0,10,2048).reshape(-1,1)

out = model(ts)

# plotting the function from t_0=0 to t_1=10
plt.plot(ts.detach().numpy(),out.detach().numpy())
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$\theta(t)$')
plt.savefig('pendu.png')
#plt.show()