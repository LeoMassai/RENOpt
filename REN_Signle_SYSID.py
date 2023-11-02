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
filepath = pjoin(folderpath, 'tanklin.mat')
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

n = np.shape(dExp[0, 0])[0]  # input dimensions

p = np.shape(yExp[0, 0])[0]  # output dimensions

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

epochs = 300
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    if epoch == epochs - epochs / 3:
        # learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        yRENm = torch.randn(p, t_end + 1, device=device, dtype=dtype)
        xi = torch.randn(n_xi)
        d = torch.from_numpy(dExp[0, exp]).float().to(device)
        for t in range(1, t_end):
            yRENm[:, t], xi = RENsys(t, d[:, t - 1], xi)
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

yRENm = torch.randn(p, t_end + 1, device=device, dtype=dtype)
xi = torch.randn(n_xi)
d = torch.from_numpy(dExp[0, exp]).float().to(device)


for t in range(1, t_end):
    yRENm[:, t], xi = RENsys(t, d[:, t - 1], xi)
y = torch.from_numpy(yExp[0, exp]).float().to(device)
y = y.squeeze()


plt.figure('1')
plt.plot(yRENm[0, :].detach().numpy(), label='REN')
plt.plot(y[0, :].detach().numpy(), label='y train')
plt.title("training")
plt.legend()
plt.show()

plt.figure('1')
plt.plot(d[2, :].detach().numpy(), label='REN')
plt.title("input")
plt.legend()
plt.show()
