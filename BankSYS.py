from models import RENR, REN
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from os.path import dirname, join as pjoin
import torch
from torch import nn


seed = 1
torch.manual_seed(seed)

dtype = torch.float
device = torch.device("cpu")

t_end = 30
nExp = 2
w = 100

x0 = torch.randn(4, t_end)  # initial conditions

# Create a tensor of dimensions 4x200x2 filled with random values between -3 and 3
u = 6 * torch.rand(4, 200, 2) - 3  # Random values between -3 and 3

# Specify the step after which you want to change the values along the second dimension
change_step = 20

# Generate new sets of random values for the first and third dimensions between -3 and 3
new_random_values_first = 6 * torch.rand(4, 2) - 3  # Random values between -3 and 3
new_random_values_third = 6 * torch.rand(4, 2) - 3  # Random values between -3 and 3

# Loop through the tensor and change values along the second dimension after 20 steps
# Also, change values along the first and third dimensions independently
for i in range(4):
    for j in range(200):
        if j % change_step == 0:
            new_random_values_first[i] = 6 * torch.rand(2) - 3  # New values for the first dimension
            new_random_values_third[i] = 6 * torch.rand(2) - 3  # New values for the third dimension
        u[i, j, 0] = new_random_values_first[i, 0]  # Change values in the first dimension
        u[i, j, 1] = new_random_values_third[i, 1]  # Change values in the third dimension

u = 2 * u

d = u[:, :, 0]
plt.figure('1')
plt.plot(d[0, :].detach().numpy(), label='REN')
plt.plot(d[1, :].detach().numpy(), label='REN')
plt.plot(d[2, :].detach().numpy(), label='REN')
plt.plot(d[3, :].detach().numpy(), label='REN')
plt.title("training")
plt.show()

yExp = torch.zeros(4, t_end, nExp).float()  # outputs / states

A = torch.tensor([[0, .3, 0, .7], [0, 0, .1, .9], [.3, .3, 0, .4], [1, 0, 0, 0]]).float()

for k in range(nExp):
    yExp[:, 0, k] = 3 * torch.abs(torch.randn(4))
    for t in range(1, t_end):
        yExp[:, t, k] = (torch.minimum(
            torch.maximum((torch.matmul(A.T, yExp[:, t - 1, k]) + u[:, t - 1, k]).unsqueeze(1), torch.zeros(4, 1)),
            w * torch.ones(4, 1))).squeeze()


plt.figure('1')
plt.plot(yExp[0, :, 0].detach().numpy(), label='Sample')
plt.plot(yExp[1, :, 0].detach().numpy(), label='Sample')
plt.plot(yExp[2, :, 0].detach().numpy(), label='Sample')
plt.plot(yExp[3, :, 0].detach().numpy(), label='Sample')
plt.title("Dynamics")
plt.legend()
plt.show()

n = u.size(0)  # input dimensions

p = yExp.size(0)  # output dimensions

n_xi = 40
# nel paper n, numero di stati
l = 20  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

RENsys = REN(n, p, n_xi, l)

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

epochs = 100
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    if epoch == epochs - epochs / 3:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp):
        yRENm = torch.randn(p, t_end, device=device, dtype=dtype)
        xi = torch.randn(n_xi)
        for t in range(1, t_end):
            yRENm[:, t], xi = RENsys(t, u[:, t - 1, exp], xi)
        y = yExp[:, :, exp]
        y = y.squeeze()
        loss = loss + MSE(yRENm[:, 10:t_end], y[:, 10:t_end])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizer.step()
    RENsys.set_model_param()

    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    #print(f"Gamma1: {RENsys.sg ** 2}")
    LOSS[epoch] = loss

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

yRENm = torch.randn(p, t_end + 1, device=device, dtype=dtype)
xi = torch.randn(n_xi)

exp = 0

for t in range(1, t_end):
    yRENm[:, t], xi = RENsys(t, u[:, t - 1, exp], xi)
y = yExp[:, :, exp]
y = y.squeeze()

plt.figure('1')
plt.plot(yRENm[3, :].detach().numpy(), label='REN')
plt.plot(y[3, :].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()
