# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# from model import REN
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os
from os.path import dirname, join as pjoin
import torch
from torch import nn
from models import RNNModel, LSTModel, OptNetf

# Import Data

N = 100  # number of samples
L = 400  # length of each sample (number of values for each sine wave)
T = 20  # width of the wave
x = np.empty((N, L, 1), np.float32)  # instantiate empty array
x[:, :, 0] = np.arange(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
y = np.sin(x / 1.0 / T).astype(np.float32)

ut = torch.from_numpy(x)
yt = torch.from_numpy(y)

# u = u.double()
# y = y.double()

# t = np.arange(0, np.size(ut, 1) * Ts, Ts)

seed = 1
torch.manual_seed(seed)

idd = 1
hdd = 11
ldd = 2
odd = 1

net = torch.nn.Sequential(
    LSTModel(idd, hdd, ldd, odd),
    # OptNetf(y.size(1))
)

MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer.zero_grad()


epochs = 4200
lossp = np.zeros(epochs)
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    ym = net(ut)
    loss = MSE(ym, yt)
    loss.backward()
    optimizer.step()
    lossp[epoch] = loss
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")

# Simulate one forward

# ymv = net(us.transpose(1, 2))
# #
plt.figure()
plt.plot(yt[4, :, 0].detach().numpy())
plt.plot(ym[4, :, 0].detach().numpy())
#
plt.show()
#
#
plt.figure()
plt.plot(lossp)
plt.show()
