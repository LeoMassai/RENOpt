from models import RENR
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
filepath = pjoin(folderpath, 'fissocarrelli100.mat')
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

n = 3  # input dimensions

p = 6  # output dimensions

n_xi = 11
# nel paper n, numero di stati
l = 10  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

RENsys = RENR(n, p, n_xi, l)

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].shape[1] - 1

epochs = 100
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    if epoch == epochs - epochs / 3:
        # learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        yRENm = torch.randn(6, t_end + 1, device=device, dtype=dtype)
        xi = torch.randn(n_xi)
        d = torch.from_numpy(dExp[0, exp]).float().to(device)
        for t in range(1, t_end):
            u = torch.tensor([d[2, t], d[7, t], d[10, t]])
            yRENm[:, t], xi = RENsys(t, u, xi)
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        y = y.squeeze()
        loss = loss + MSE(yRENm[:, 10:yRENm.size(1)], y[:, 10:t_end + 1])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizer.step()
    RENsys.set_model_param()

    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma1: {RENsys.sg ** 2}")
    LOSS[epoch] = loss

plt.figure('3')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('1')
plt.plot(yRENm[4, :].detach().numpy(), label='REN')
plt.plot(y[4, :].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()
