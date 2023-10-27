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
filepath = pjoin(folderpath, 'datasetfisso2.mat')
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
m = 5  # nel paper p, numero di uscite
p = 1

n_xi = 20
n_xi2 = 13
# nel paper n, numero di stati
l = 30  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
l2 = 12
l2 = 16
RENsys = doubleRENg(n, m, p, n_xi, n_xi2, l, l2)

# Define the system

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
        yREN = torch.zeros(t_end + 1, RENsys.p, device=device, dtype=dtype)
        xi = torch.zeros(RENsys.n_xi, device=device, dtype=dtype)
        xi2 = torch.zeros(RENsys.n_xi2, device=device, dtype=dtype)
        # Simulate one forward
        u = torch.from_numpy(uExp[0, exp]).float().to(device)
        for t in range(t_end):
            yREN[t, :], xi, xi2 = RENsys(t, u[:, t], xi, xi2)
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        loss = loss + MSE(yREN[10:yREN.size(0)], y[10:y.size(0)])
        # ignorare da loss effetto condizione iniziale
    loss = loss / nExp
    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma1: {RENsys.r1.gamma}")
    print(f"GammaProd: {RENsys.r2.gamma*RENsys.r1.gamma}")
    LOSS[epoch] = loss
    RENsys.set_model_param()

# training set
t_end = yExp[0, 0].size - 1
yREN = torch.empty(t_end + 1, RENsys.p, device=device, dtype=dtype)
xi = torch.zeros(RENsys.n_xi, device=device, dtype=dtype)
xi2 = torch.zeros(RENsys.n_xi2, device=device, dtype=dtype)
# Simulate one forward
u = torch.from_numpy(uExp[0, 0]).float().to(device)
for t in range(t_end):
    yREN[t, :], xi, xi2 = RENsys(t, u[:, t], xi, xi2)
y = torch.from_numpy(yExp[0, 0]).float().to(device)
loss = MSE(yREN, y)

# validation
t_end = yExp_val[0, 0].size - 1
yREN_val = torch.empty(t_end + 1, RENsys.p, device=device, dtype=dtype)
xi = torch.zeros(RENsys.n_xi, device=device, dtype=dtype)
xi2 = torch.zeros(RENsys.n_xi2, device=device, dtype=dtype)
# Simulate one forward
u = torch.from_numpy(uExp_val[0, 0]).float().to(device)
for t in range(t_end):
    yREN_val[t, :], xi, xi2 = RENsys(t, u[:, t], xi, xi2)
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
