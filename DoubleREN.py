from models import REN
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os
from os.path import dirname, join as pjoin
import torch
from torch import nn
from models import RNNModel, OptNet, OptNetf, RENRG, doubleREN

plt.close('all')

# Import Data


folderpath = os.getcwd()

filepath = pjoin(folderpath, 'datasetb.mat')

data = scipy.io.loadmat(filepath)

u, y, Ts = torch.from_numpy(data['u']).float(), torch.from_numpy(
    data['y']).float(), data['Ts'].item()

u = (u - u.min()) / (u.max() - u.min())
#
y = (y - y.min()) / (y.max() - y.min())

nExp = 20

ut = u[:nExp, :, :]
yt = y[:nExp, :]

us = u[nExp + 1:, :, :]
ys = y[nExp + 1:, :]

# u = u.double()
# y = y.double()

t = np.arange(0, np.size(ut, 1) * Ts, Ts)

seed = 1
torch.manual_seed(seed)

idd = u.size(1)
hdd = 30
ldd = 3
odd = 1

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
RENsys = doubleREN(n, m, p, n_xi, n_xi2, l, l2)

params = list(RENsys.r1.parameters()) + list(RENsys.r2.parameters()) + list(RENsys.parameters())

MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-3
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = y.size(1)
epochs = 60
lossp = np.zeros(epochs)

for epoch in range(epochs):

    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp):
        yREN = torch.zeros(t_end, RENsys.p)
        xi = torch.zeros(RENsys.n_xi)
        xi2 = torch.zeros(RENsys.n_xi2)
        # Simulate one forward
        for t in range(t_end):
            yREN[t, :], xi, xi2 = RENsys(t, ut[exp, :, t], xi, xi2)
        yRENl = yREN.squeeze()
        loss = loss + MSE(yRENl[10:t_end], yt[exp, 10:t_end])
    loss = loss / nExp
    # loss.backward(retain_graph=True)
    loss.backward(retain_graph=True)
    optimizer.step()
    lossp[epoch] = loss
    RENsys.set_model_param()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma1: {RENsys.r1.gamma}")
    print(f"Gamma2: {RENsys.r2.gamma}")

# Training
exp = 12
xi = torch.zeros(RENsys.n_xi)
xi2 = torch.zeros(RENsys.n_xi2)
yRENt = torch.zeros(t_end, RENsys.p)
for t in range(t_end):
    yRENt[t, :], xi, xi2 = RENsys(t, ut[exp, :, t], xi, xi2)

plt.figure()
plt.plot(yt[12, :].detach().numpy())
plt.plot(yRENt.detach().numpy())
plt.show()

# ex = 13
# yREN_val = torch.zeros(t_end, RENsys.m)
# for t in range(t_end):
#     yREN_val[t, :], xi = RENsys(t, us[ex, :, t], xi)
# yREN_val = yREN_val.squeeze()
# lossVal = MSE(yREN, ys[ex, :])
#
# plt.figure()
# plt.plot(ys[ex, :].detach().numpy(), label='Y_val')
# plt.plot(yREN.detach().numpy(), label='Opt')
# plt.title("validation")
# plt.legend()
# plt.show()

# #
# plt.figure()
# plt.plot(ys[13, :].detach().numpy())
# plt.plot(ymv[13, :].detach().numpy())
# #
# plt.show()
#
#
plt.figure()
plt.plot(lossp)
plt.show()
