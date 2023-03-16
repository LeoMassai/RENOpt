# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#from model import REN
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os
from os.path import dirname, join as pjoin
import torch
from torch import nn
from models import RNNModel, OptNet, OptNetf

# Import Data


folderpath = os.getcwd()

filepath = pjoin(folderpath, 'datasetb.mat')

data = scipy.io.loadmat(filepath)

u, y, Ts = torch.from_numpy(data['u']).float(), torch.from_numpy(
    data['y']).float(), data['Ts'].item()

# u = (u - u.min()) / (u.max() - u.min())
#
# y = (y - y.min()) / (y.max() - y.min())

ut=u[:70,: ,:]
yt=y[:70,:]

us=u[71:,: ,:]
ys=y[71:,:]

# u = u.double()
# y = y.double()

t = np.arange(0, np.size(ut, 1) * Ts, Ts)

seed = 1
torch.manual_seed(seed)

idd = u.size(1)
hdd = 30
ldd = 3
odd = 1

net = torch.nn.Sequential(
    RNNModel(idd, hdd, ldd, odd),
    OptNetf(y.size(1))
)

MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = Ts
epochs = 700
lossp = np.zeros(epochs)
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    ym = net(ut.transpose(1, 2))
    ym = torch.squeeze(ym)
    loss = MSE(ym, yt)
    loss.backward()
    optimizer.step()
    lossp[epoch] = loss
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")

# Simulate one forward

ymv = net(us.transpose(1, 2))
# #
plt.figure()
plt.plot(ys[13, :].detach().numpy())
plt.plot(ymv[13, :].detach().numpy())
#
plt.show()
#
#
plt.figure()
plt.plot(lossp)
plt.show()
