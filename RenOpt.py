from models import REN
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os
from os.path import dirname, join as pjoin
import torch
from torch import nn
from models import RNNModel, OptNet, OptNetf

plt.close('all')

# Import Data


folderpath = os.getcwd()

filepath = pjoin(folderpath, 'datasetb.mat')

data = scipy.io.loadmat(filepath)

u, y, Ts = torch.from_numpy(data['u']).float(), torch.from_numpy(
    data['y']).float(), data['Ts'].item()

# u = (u - u.min()) / (u.max() - u.min())
#
# y = (y - y.min()) / (y.max() - y.min())

nExp = 70

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

n_xi = 25  # nel paper n, numero di stati
l = 30  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the REN

RENsys = REN(idd, odd, n_xi, l)
Optl = OptNetf(1)

MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
params = list(RENsys.parameters()) + list(Optl.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
optimizer.zero_grad()

t_end = y.size(1)
epochs = 300
lossp = np.zeros(epochs)

for epoch in range(epochs):
    if epoch == epochs - epochs / 3:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(params, lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp):
        yREN = torch.zeros(t_end, RENsys.m)
        yOpt = torch.zeros(t_end, RENsys.m)
        xi = torch.zeros(RENsys.n_xi)
        # Simulate one forward
        for t in range(t_end):
            yREN[t], xi = RENsys(t, ut[exp, :, t], xi)
            yOpt[t] = Optl(yREN[t])
        yOpt = yOpt.squeeze()
        loss = loss + MSE(yOpt, yt[exp, :])
    loss = loss / nExp
    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    lossp[epoch] = loss
    RENsys.set_model_param()

# validation

ex = 9
yREN_val = torch.zeros(t_end, RENsys.m)
yOpt_val = torch.zeros(t_end, RENsys.m)
for t in range(t_end):
    yREN_val[t, :], xi = RENsys(t, us[ex, :, t], xi)
    yOpt_val[t] = Optl(yREN_val[t])
yOpt_val = yOpt_val.squeeze()
lossVal = MSE(yOpt_val, ys[ex, :])

plt.figure()
plt.plot(ys[ex, :].detach().numpy(), label='Y_val')
plt.plot(yOpt_val.detach().numpy(), label='Opt')
plt.title("validation")
plt.legend()
plt.show()

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
