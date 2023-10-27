from models import pizzicottina4
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

seed = 1
torch.manual_seed(seed)

n = torch.tensor([2, 2, 1, 1])  # input dimensions

p = torch.tensor([1, 1, 1, 1])  # output dimensions

n_xi = np.array([4, 4, 4, 4])
# nel paper n, numero di stati
l = 2*np.array([5, 5, 5, 5])  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

M = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
M = M.float()
N = 4

RENsys = pizzicottina4(N, M, n, p, n_xi, l)

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].shape[1] - 1

epochs = 250
LOSS = np.zeros(epochs)
loss = 10

for epoch in range(epochs):
    if loss < 1.0e-2:
        learning_rate = 1.0e-2
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        xi = []
        yRENm = torch.randn(4, t_end + 1, device=device, dtype=dtype)
        yRENma = torch.randn(4, t_end + 1, device=device, dtype=dtype)
        for j in range(N):
            xi.append(torch.randn(RENsys.r[j].n_xi, device=device, dtype=dtype))

        d = torch.from_numpy(dExp[0, exp]).float().to(device)
        xi = torch.cat(xi)
        for t in range(1, t_end):
            yRENm[:, t], xi, yRENma[:, t] = RENsys(t, yRENm[:, t - 1], d[:, t - 1], xi)
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        y = y.squeeze()
        loss = loss + MSE(yRENma[:, 10:yRENma.size(1)], y[:, 10:t_end + 1])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizer.step()
    RENsys.set_model_param()

    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma1: {RENsys.r[0].gamma}, {RENsys.q}")
    print(f"Gamma2: {RENsys.r[1].gamma}")
    print(f"Gamma3: {RENsys.r[2].gamma}")
    LOSS[epoch] = loss

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

xi = []
yRENm = torch.zeros(4, t_end + 1, device=device, dtype=dtype)
yRENma = torch.zeros(4, t_end + 1, device=device, dtype=dtype)
for j in range(N):
    xi.append(torch.zeros(RENsys.r[j].n_xi, device=device, dtype=dtype))
d = torch.from_numpy(dExp[0, 1]).float().to(device)
xi = torch.cat(xi)
for t in range(1, t_end):
    yRENm[:, t], xi, yRENma[:, t] = RENsys(t, yRENm[:, t - 1], d[:, t - 1], xi)
y = torch.from_numpy(yExp[0, 1]).float().to(device)
y = y.squeeze()

plt.figure('1')
plt.plot(yRENma[0, 0:t_end].detach().numpy(), label='REN')
plt.plot(y[0, 0:t_end].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()
