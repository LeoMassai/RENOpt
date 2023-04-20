# -*- coding: utf-8 -*-

from models import doubleRENg, RENR2
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from os.path import dirname, join as pjoin
import torch
from torch import nn

dtype = torch.float
device = torch.device("cpu")

plt.close('all')
# Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'datasetsiso.mat')
data = scipy.io.loadmat(filepath)

uExp, yExp, uExp_val, yExp_val, Ts = data['uExp'], data['yExp'], data['uExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(uExp[0, 0], 1) * Ts, Ts)

# plt.plot(t, yExp[0,-1])
# plt.show()

seed = 1
torch.manual_seed(seed)

n = 1  # nel paper m, numero di ingressi
p = 1

n_xi = 20
n_xi2 = 13
# nel paper n, numero di stati
l = 30  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE
l2 = 16
RENsys = doubleRENg(n, p, n_xi, n_xi2, l, l2)

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].size - 1

epochs = 90
LOSS = np.zeros(epochs)
for epoch in range(epochs):
    if epoch == epochs - epochs / 3:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        yREN1 = torch.zeros(t_end + 1, RENsys.p, device=device, dtype=dtype)
        yREN2 = torch.zeros(t_end + 1, RENsys.n, device=device, dtype=dtype)
        yREN = torch.zeros(t_end + 1, RENsys.p, device=device, dtype=dtype)
        xi = torch.zeros(RENsys.n_xi, device=device, dtype=dtype)
        xi2 = torch.zeros(RENsys.n_xi2, device=device, dtype=dtype)
        yREN1[0, :], yREN2[0, :], xi, xi2 = RENsys(0, torch.randn(1, n, device=device, dtype=dtype), xi, xi2)

        # Simulate one forward
        u = torch.from_numpy(uExp[0, exp]).float().to(device)
        for t in range(1, t_end):
            ul = u[:, t] - yREN2[t - 1, :]
            yREN1[t, :], yREN2[t, :], xi, xi2 = RENsys(t, ul, xi, xi2)
            yREN[t, :] = yREN1[t, :]
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        loss = loss + MSE(yREN[10:yREN.size(0)], y[10:y.size(0)])
        # ignorare da loss effetto condizione iniziale
    loss = loss / nExp
    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma1: {RENsys.r1.gamma}")
    print(f"Gamma1: {RENsys.r2.gamma}")
    print(f"GammaProd: {RENsys.r2.gamma * RENsys.r1.gamma}")
    LOSS[epoch] = loss
    RENsys.set_model_param()

# training set
t_end = yExp[0, 0].size - 1
yREN1 = torch.zeros(t_end + 1, RENsys.p, device=device, dtype=dtype)
yREN2 = torch.zeros(t_end + 1, RENsys.n, device=device, dtype=dtype)
yREN = torch.zeros(t_end + 1, RENsys.p, device=device, dtype=dtype)
xi = torch.zeros(RENsys.n_xi, device=device, dtype=dtype)
xi2 = torch.zeros(RENsys.n_xi2, device=device, dtype=dtype)
yREN1[0, :], yREN2[0, :], xi, xi2 = RENsys(0, torch.randn(1, n, device=device, dtype=dtype), xi, xi2)

# Simulate one forward
u = torch.from_numpy(uExp[0, 0]).float().to(device)
for t in range(t_end):
    ul = u[:, t] - yREN2[t - 1, :]
    yREN1[t, :], yREN2[t, :], xi, xi2 = RENsys(t, ul, xi, xi2)
    yREN[t, :] = yREN1[t, :]
y = torch.from_numpy(yExp[0, 0]).float().to(device)
loss = MSE(yREN, y)

# validation
t_end = yExp_val[0, 0].size - 1
yREN1_val = torch.zeros(t_end + 1, RENsys.p, device=device, dtype=dtype)
yREN2_val = torch.zeros(t_end + 1, RENsys.n, device=device, dtype=dtype)
yREN_val = torch.zeros(t_end + 1, RENsys.p, device=device, dtype=dtype)
xi = torch.zeros(RENsys.n_xi, device=device, dtype=dtype)
xi2 = torch.zeros(RENsys.n_xi2, device=device, dtype=dtype)
yREN1_val[0, :], yREN2_val[0, :], xi, xi2 = RENsys(0, torch.randn(1, n, device=device, dtype=dtype), xi, xi2)
# Simulate one forward
u = torch.from_numpy(uExp_val[0, 0]).float().to(device)
for t in range(t_end):
    ul = u[:, t] - yREN2_val[t - 1, :]
    yREN1_val[t, :], yREN2_val[t, :], xi, xi2 = RENsys(t, ul, xi, xi2)
    yREN_val[t, :] = yREN1_val[t, :]
y_val = torch.from_numpy(yExp_val[0, 0]).float().to(device)
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
