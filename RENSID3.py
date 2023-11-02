from models import pizzicottina5
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
filepath = pjoin(folderpath, 'datatank.mat')
data = scipy.io.loadmat(filepath)

dExp, yExp, dExp_val, yExp_val, Ts = data['dExp'], data['yExp'], \
    data['dExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

nExp = 2

t = np.arange(0, np.size(dExp[0, 0], 1) * Ts, Ts)

# plt.plot(t, yExp[0,-1])
# plt.show()


n = torch.tensor([2, 2, 1, 1])  # input dimensions

p = torch.tensor([1, 1, 1, 1])  # output dimensions

n_xi = torch.tensor([6, 6, 6, 6])
# nel paper n, numero di stati
l = 2 * torch.tensor(
    [5, 5, 5, 5])  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

M = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
M = M.float()
N = 4
t_end = yExp[0, 0].shape[1] - 1

RENsys = pizzicottina5(N, M, n, p, n_xi, l, t_end)

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

epochs = 200
LOSS = np.zeros(epochs)


for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        d = torch.from_numpy(dExp[0, exp]).float().to(device)
        yRENm, xi, yRENma = RENsys(t, d)
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        y = y.squeeze()
        loss = loss + MSE(yRENma[:, 10:yRENma.size(1)], y[:, 10:t_end + 1])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    loss.backward(retain_graph=True)

    optimizer.step()
    RENsys.set_model_param()

    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma1: {RENsys.r[0].gamma}")
    print(f"Gamma2: {RENsys.r[1].gamma}")
    print(f"Gamma3: {RENsys.r[2].gamma}")
    LOSS[epoch] = loss

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

d = torch.from_numpy(dExp[0, 1]).float().to(device)
yRENm, xi, yRENma = RENsys(t, d)
y = torch.from_numpy(yExp[0, 1]).float().to(device)
y = y.squeeze()

plt.figure('1')
plt.plot(yRENma[0, 0:t_end].detach().numpy(), label='REN')
plt.plot(y[0, 0:t_end].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()
