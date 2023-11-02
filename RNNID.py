# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os
from os.path import dirname, join as pjoin
import torch
from torch import nn
from models import RNNModel

# Import Data


folderpath = os.getcwd()

filepath = pjoin(folderpath, 'dataset_sysID_3tanks_final.mat')

data = scipy.io.loadmat(filepath)

dExp, yExp, dExp_val, yExp_val, Ts = data['dExp'], data['yExp'], \
    data['dExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(dExp[0, 0], 1) * Ts, Ts).squeeze()

t_end = t.size

u = torch.zeros(nExp, t_end, 4)
y = torch.zeros(nExp, t_end, 3)

for j in range(nExp):
    u[j, :, :] = (torch.from_numpy(dExp[0, j])).T
    y[j, :, :] = (torch.from_numpy(yExp[0, j])).T

seed = 1
torch.manual_seed(seed)

idd = dExp[0, 0].shape[0]
hdd = 10
ldd = 5
odd = yExp[0, 0].shape[0]

RNN = RNNModel(idd, hdd, ldd, odd)
MSE = nn.MSELoss()

pytorch_total_params_RNN = sum(p.numel() for p in RNN.parameters() if p.requires_grad)

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
optimizer.zero_grad()

epochs = 500
LOSS = np.zeros(epochs)
lossp = np.zeros(epochs)
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    yRNN = RNN(u)
    yRNN = torch.squeeze(yRNN)
    loss = MSE(yRNN, y)
    loss.backward()
    optimizer.step()
    lossp[epoch] = loss
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('1')
plt.plot(yRNN[0, 0:t_end, 2].detach().numpy(), label='RNN')
plt.plot(y[0, 0:t_end, 2].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()

# %% BANK ID


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
from models import RNNModel

seed = 1
torch.manual_seed(seed)

dtype = torch.float
device = torch.device("cpu")

t_end = 200
nExp = 2
w = 14

x0 = torch.randn(4, t_end)  # initial conditions

# Create a tensor of dimensions 4x200x2 filled with random values between -3 and 3
u = 6 * torch.rand(2, 200, 4) - 3  # Random values between -3 and 3

# Specify the step after which you want to change the values along the second dimension
change_step = 20

# Generate new sets of random values for the first and third dimensions between -3 and 3
new_random_values_first = 6 * torch.rand(4, 2) - 3  # Random values between -3 and 3
new_random_values_third = 6 * torch.rand(4, 2) - 3  # Random values between -3 and 3

# Loop through the tensor and change values along the second dimension after 20 steps
# Also, change values along the first and third dimensions independently
for i in range(4):
    for j in range(t_end):
        if j % change_step == 0:
            new_random_values_first[i] = 6 * torch.rand(2) - 3  # New values for the first dimension
            new_random_values_third[i] = 6 * torch.rand(2) - 3  # New values for the third dimension
        u[0, j, i] = new_random_values_first[i, 0]  # Change values in the first dimension
        u[1, j, i] = new_random_values_third[i, 1]  # Change values in the third dimension

u = 2 * u

d = u[0, :, :]
plt.figure('1')
plt.plot(d[0, :].detach().numpy(), label='REN')
plt.plot(d[1, :].detach().numpy(), label='REN')
plt.plot(d[2, :].detach().numpy(), label='REN')
plt.plot(d[3, :].detach().numpy(), label='REN')
plt.title("training")
plt.show()

yExp = torch.zeros(nExp, t_end, 4).float()  # outputs / states

A = torch.tensor([[0, .3, 0, .7], [0, 0, .1, .9], [.3, .3, 0, .4], [1, 0, 0, 0]]).float()

for k in range(nExp):
    yExp[k, 0, :] = 3 * torch.abs(torch.randn(4))
    for t in range(1, t_end):
        yExp[k, t, :] = (torch.minimum(
            torch.maximum((torch.matmul(A.T, yExp[k, t - 1, :]) + u[k, t - 1, :]).unsqueeze(1), torch.zeros(4, 1)),
            w * torch.ones(4, 1))).squeeze()

plt.figure('1')
plt.plot(yExp[0, :, 0].detach().numpy(), label='Sample')
plt.plot(yExp[0, :, 1].detach().numpy(), label='Sample')
plt.plot(yExp[0, :, 2].detach().numpy(), label='Sample')
plt.plot(yExp[0, :, 3].detach().numpy(), label='Sample')
plt.title("Dynamics")
plt.legend()
plt.show()

n = u.size(0)  # input dimensions

p = yExp.size(0)  # output dimensions

seed = 1
torch.manual_seed(seed)

idd = 4
hdd = 10
ldd = 5
odd = 4

RNN = RNNModel(idd, hdd, ldd, odd)
MSE = nn.MSELoss()

pytorch_total_params_RNN = sum(p.numel() for p in RNN.parameters() if p.requires_grad)

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
optimizer.zero_grad()

epochs = 1200
LOSS = np.zeros(epochs)
lossp = np.zeros(epochs)
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    yRNN = RNN(u)
    yRNN = torch.squeeze(yRNN)
    loss = MSE(yRNN, yExp)
    loss.backward()
    optimizer.step()
    lossp[epoch] = loss
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('1')
plt.plot(yRNN[0, 0:t_end, 2].detach().numpy(), label='RNN')
plt.plot(yExp[0, 0:t_end, 2].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()

# Training
nt = 2
yt = torch.abs(3 * torch.rand(nt, t_end, 4))
ut = 6 * torch.rand(nt, 200, 4) - 3  # Random values between -3 and 3


for i in range(4):
    for j in range(t_end):
        if j % change_step == 0:
            new_random_values_first[i] = 6 * torch.rand(2) - 3  # New values for the first dimension
            new_random_values_third[i] = 6 * torch.rand(2) - 3  # New values for the third dimension
        ut[0, j, i] = new_random_values_first[i, 0]  # Change values in the first dimension
        ut[1, j, i] = new_random_values_third[i, 1]  # Change values in the third dimension

ut = 2 * ut

for k in range(nt):
    for t in range(1, t_end):
        yt[k, t, :] = (torch.minimum(
            torch.maximum((torch.matmul(A.T, yt[k, t - 1, :]) + ut[k, t - 1, :]).unsqueeze(1), torch.zeros(4, 1)),
            w * torch.ones(4, 1))).squeeze()

ytRNN = RNN(ut)

plt.figure('1')
plt.plot(ytRNN[0, 0:t_end, 0].detach().numpy(), label='RNN')
plt.plot(yt[0, 0:t_end, 0].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()
