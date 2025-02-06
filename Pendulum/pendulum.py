import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#form torchvision import datasets, transform
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt

#***********************************************************
# A PINN to solve the pendulum equation (P. Waldrop)
# L * d^2u/dt^2 + g * sin(u) = 0
#***********************************************************

# length of pendulem
ell = 2.0
# acceleration due to gravity
g   = 9.81
# initial energy based on initial conditions (hard coded)
ei = g*ell + 0.5*(ell)**2


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
    
# loss function
def auto_loss(model,ts):
    # u(t) from the network
    thetas = model.forward(ts)

    # u_dot(t) first derivative
    theta_dot = torch.autograd.grad(thetas,ts, torch.ones_like(thetas), create_graph=True)[0]

    # u_dotdot(t) second derivative
    theta_dotdot = torch.autograd.grad(theta_dot,ts, torch.ones_like(theta_dot), create_graph=True)[0]

    # differential equation
    loss1 = ell*theta_dotdot + g * torch.sin(thetas)
    loss1 = torch.mean(torch.abs(loss1)**2)

    # initial position u(t=0)=pi/2
    loss2 = torch.abs(model(torch.tensor([[0.0]])) - 3.14159/2)**2

    # initial velocity u_dot(t=0)=1
    zer = torch.tensor([[0.0]])
    zer.requires_grad = True
    thes = model.forward(zer)
    thes_dot = torch.autograd.grad(thes, zer, torch.ones_like(thes), create_graph=True)[0]
    loss3 = torch.abs(thes_dot + 1.0)**2

    # Conservation of energy 
    energy = g*ell*(1-torch.cos(thetas)) + 0.5*(ell*theta_dot)**2
    
    loss4 = torch.mean((energy - ei)**2)

    loss = loss1 + loss2[0,0] + loss3 + 0.01*loss4

    #print(f'{loss.item()}')
    return loss

model = Net(4,256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training
for epoch in range(100000):
    ts = torch.randn(512,1)*10
    ts.requires_grad = True
    optimizer.zero_grad()

    # calculating loss function
    loss = auto_loss(model, ts)
    # periodically printing current loss
    if epoch%10000 == 0:
        print(f'{loss.item()}')

    loss.backward()
    optimizer.step()

print('Finished training!')

torch.save(model.state_dict(), 'pendulum.pt')

# times to plot
ts = torch.linspace(0,10.0,2048).reshape((2048,1))
ts.requires_grad = True
# u(t) final evaluations
out = model(ts)

# plotting the function from t_0=0 to t_1=10
plt.plot(ts.detach().numpy(),out.detach().numpy())
plt.show()