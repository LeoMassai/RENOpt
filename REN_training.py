# -*- coding: utf-8 -*-

from models import REN
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os
from os.path import dirname, join as pjoin
import torch
from torch import nn

plt.close('all')
# Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'dataset.mat')
data = scipy.io.loadmat(filepath)

uExp, yExp, uExp_val, yExp_val, Ts = data['uExp'], data['yExp'], data['uExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(uExp[0, 0], 1) * Ts, Ts)

# plt.plot(t, yExp[0,-1])
# plt.show()

seed = 1
torch.manual_seed(seed)

n = 2  # nel paper m, numero di ingressi
n_xi = 20  # nel paper n, numero di stati
l = 20  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
m = 1  # nel paper p, numero di uscite

# Define the system
RENsys = REN(n, m, n_xi, l)

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].size - 1

epochs = 300
LOSS = np.zeros(epochs)
for epoch in range(epochs):
    if epoch == epochs - epochs / 3:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        yREN = torch.zeros(t_end + 1, RENsys.m)
        xi = torch.zeros(RENsys.n_xi)
        # Simulate one forward
        u = torch.from_numpy(uExp[0, exp]).float()
        for t in range(t_end):
            yREN[t, :], xi = RENsys(t, u[:, t], xi)
        y = torch.from_numpy(yExp[0, exp]).float()
        loss = loss + MSE(yREN[10:yREN.size(0)], y[10:y.size(0)])
        # ignorare da loss effetto condizione iniziale
    loss = loss / nExp
    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss
    RENsys.set_model_param()

# training set
t_end = yExp[0, 0].size - 1
yREN = torch.empty(t_end + 1, RENsys.m)
xi = torch.zeros(RENsys.n_xi)
# Simulate one forward
u = torch.from_numpy(uExp[0, 0]).float()
for t in range(t_end):
    yREN[t, :], xi = RENsys(t, u[:, t], xi)
y = torch.from_numpy(yExp[0, 0]).float()
loss = MSE(yREN, y)

# validation
t_end = yExp_val[0, 0].size - 1
yREN_val = torch.empty(t_end + 1, RENsys.m)
xi = torch.zeros(RENsys.n_xi)
# Simulate one forward
u = torch.from_numpy(uExp_val[0, 0]).float()
for t in range(t_end):
    yREN_val[t, :], xi = RENsys(t, u[:, t], xi)
y_val = torch.from_numpy(yExp_val[0, 0]).float()
lossVal = MSE(yREN_val, y_val)

print(f"Loss in validation: {lossVal}")

plt.figure('1')
plt.plot(yREN.detach().numpy(), label='REN')
plt.plot(y.detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()

plt.figure('2')
plt.plot(yREN_val.detach().numpy(), label='REN')
plt.plot(y_val.detach().numpy(), label='y val')
plt.title("validation")
plt.legend()
plt.show()

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

print('ciao')
