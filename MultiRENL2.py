# -*- coding: utf-8 -*-

from models import doubleRENg, RENR2, multiRENL2, RENRG, ThreeRENL2GershgorinShurMIMO
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

uExp, yExp, uExp_val, yExp_val, Ts = data['uExp'], data['yExp'],\
    data['uExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(uExp[0, 0], 1) * Ts, Ts)

# plt.plot(t, yExp[0,-1])
# plt.show()

seed = 1
torch.manual_seed(seed)

n = torch.tensor([1, 2, 1])
# nel paper m, numero di ingressi
p = torch.tensor([2, 1, 1])

n_xi = np.array([4, 4, 4])
# nel paper n, numero di stati
l = np.array([3, 3, 3])  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

M = torch.tensor([[0, 0, -1], [1, 0, 0], [0, 1, 0]])
M = M.float()
N = M.size(dim=1)

RENsys = ThreeRENL2GershgorinShurMIMO(N, M, n, p, n_xi, l)

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].size - 1

epochs = 40
LOSS = np.zeros(epochs)
for epoch in range(epochs):
    if epoch == epochs - epochs / 3:
        # learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        xi = []
        yRENm = torch.zeros(4, t_end + 1, device=device, dtype=dtype)
        yREN = torch.zeros(t_end + 1, device=device, dtype=dtype)
        for j in range(N):
            xi.append(torch.zeros(RENsys.r[j].n_xi, device=device, dtype=dtype))

        # Simulate one forward
        u = torch.from_numpy(uExp[0, exp]).float().to(device)
        for t in range(1, t_end):
            yRENm[:, t], xi = RENsys(t, yRENm[3, t - 1], u[:, t], xi)
            yREN[t] = yRENm[2, t]
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        y = y.squeeze()
        loss = loss + MSE(yREN[10:yREN.size(0)], y[10:y.size(0)])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizer.step()
    RENsys.set_model_param()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma1: {RENsys.r[0].gamma}")
    print(f"Gamma2: {RENsys.r[1].gamma}")
    print(f"Gamma3: {RENsys.r[2].gamma}")
    print(f"GammaProd: {RENsys.r[0].gamma * RENsys.r[1].gamma * RENsys.r[2].gamma}")
    LOSS[epoch] = loss

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

# training

xi = []
for j in range(N):
    xi.append(torch.zeros(RENsys.r[j].n_xi, device=device, dtype=dtype))
u = torch.from_numpy(uExp[0, 4]).float().to(device)
for t in range(1, t_end):
    yRENm[:, t], xi = RENsys(t, yRENm[3, t - 1], u[:, t], xi)
    yREN[t] = yRENm[2, t]
y = torch.from_numpy(yExp[0, 4]).float().to(device)
y = y.squeeze()

plt.figure('1')
plt.plot(yREN.detach().numpy(), label='REN')
plt.plot(y.detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()

# validation
t_end = yExp_val[0, 0].size - 1
xi = []
for j in range(N):
    xi.append(torch.zeros(RENsys.r[j].n_xi, device=device, dtype=dtype))
yRENm_val = torch.zeros(4, t_end + 1, device=device, dtype=dtype)
yREN_val = torch.zeros(t_end + 1, device=device, dtype=dtype)

u_val = torch.from_numpy(uExp_val[0, 0]).float().to(device)
for t in range(t_end):
    yRENm_val[:, t], xi = RENsys(t, yRENm_val[3, t - 1], u_val[:, t], xi)
    yREN_val[t] = yRENm_val[2, t]
y_val = torch.from_numpy(yExp_val[0, 0]).float().to(device)
y_val = y_val.squeeze()

plt.figure('2')
plt.plot(yREN_val.detach().numpy(), label='REN')
plt.plot(y_val.detach().numpy(), label='y val')
plt.title("validation")
plt.legend()
plt.show()
