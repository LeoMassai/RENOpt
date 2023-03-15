#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:24:52 2023

@author: leo
"""

import numpy as np  # linear algebra
import torch
import torch.nn as nn
from torch.autograd import Variable
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='relu', batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.rand(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out=self.fc(out)
        out=out.squeeze()
        return out


#Opten: Convex optimization problem as a layer
class OptNet(torch.nn.Module):
    def __init__(self, D_in):
        super(OptNet, self).__init__()
        #self.b = torch.nn.Parameter(torch.randn(D_in))
        self.G = torch.nn.Parameter(torch.randn(D_in, D_in))
        self.h = torch.nn.Parameter(torch.randn(D_in))
        G = cp.Parameter((D_in, D_in))
        h = cp.Parameter(D_in)
        z = cp.Variable(D_in)
        #b = cp.Parameter(D_in)
        x = cp.Parameter(D_in)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(z - x)), [G @ z <= h])
        self.layer = CvxpyLayer(prob, [G, h, x], [z])

    def forward(self, x):
        # when x is batched, repeat W and b
        if x.ndim == 2:
            batch_size = x.shape[0]
            return self.layer(self.G.repeat(batch_size, 1, 1),
                              self.h.repeat(batch_size, 1), x)[0]
        else:
            return self.layer(self.G, self.h, x)[0]



#Opten: Euclidean projection onto convex set
class OptNetf(torch.nn.Module):
    def __init__(self, D_in):
        super(OptNetf, self).__init__()
        #self.b = torch.nn.Parameter(torch.randn(D_in))
        G = torch.eye(D_in)
        h = 7*torch.ones(D_in)
        z = cp.Variable(D_in)
        #b = cp.Parameter(D_in)
        x = cp.Parameter(D_in)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(z - x)), [G @ z <= h])
        self.layer = CvxpyLayer(prob, [x], [z])

    def forward(self, x):
        return self.layer(x)[0]

