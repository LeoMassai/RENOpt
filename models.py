#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:24:52 2023

@author: leo
"""

import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np  # linear algebra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

dtype = torch.float
device = torch.device("cpu")


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='tanh', batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state randomly
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        out = out.squeeze()
        return out


class LSTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # One time step
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        # out = out.squeeze()
        return out


class REN(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.m = m  # nel paper p
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l) * std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi) * std))  # Y1 nel paper
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n) * std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi) * std))
        self.D21 = nn.Parameter((torch.randn(m, l) * std))
        self.D22 = nn.Parameter((torch.randn(m, n) * std))
        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n) * std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l)
        vec[0] = 1
        epsilon = torch.zeros(self.l)
        v = F.linear(xi, self.C1[0, :]) + F.linear(w,
                                                   self.D12[0, :])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon,
                                                       self.D11[i, :]) + F.linear(w, self.D12[i, :])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon,
                                                self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + \
            F.linear(w, self.D22)  # + self.bu
        return u, xi_


# Opten: Convex optimization problem as a layer
class OptNet(torch.nn.Module):
    def __init__(self, D_in):
        super(OptNet, self).__init__()
        # self.b = torch.nn.Parameter(torch.randn(D_in))
        self.G = torch.nn.Parameter(torch.randn(D_in, D_in))
        self.h = torch.nn.Parameter(torch.randn(D_in))
        G = cp.Parameter((D_in, D_in))
        h = cp.Parameter(D_in)
        z = cp.Variable(D_in)
        # b = cp.Parameter(D_in)
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


# Opten: Euclidean projection onto convex set
class OptNetf(torch.nn.Module):
    def __init__(self, D_in):
        super(OptNetf, self).__init__()
        # self.b = torch.nn.Parameter(torch.randn(D_in))
        G = torch.eye(D_in)
        h = 7 * torch.ones(D_in)
        z = cp.Variable(D_in)
        # b = cp.Parameter(D_in)
        x = cp.Parameter(D_in)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(z - x)), [G @ z <= h])
        self.layer = CvxpyLayer(prob, [x], [z])

    def forward(self, x):
        return self.layer(x)[0]


class RENR(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.m = m  # nel paper p
        self.s = np.max((n, m))  # s nel paper, dimensione di X3 Y3

        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l) * std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi) * std))  # Y1 nel paper
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n) * std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi) * std))
        self.D21 = nn.Parameter((torch.randn(m, l) * std))
        self.X3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.Y3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.sg = nn.Parameter(torch.randn(1, 1) * std)  # square root of gamma

        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n) * std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.Lq = torch.zeros(m, m)
        self.Lr = torch.zeros(n, n)
        self.D22 = torch.zeros(m, n)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        n = self.n
        m = self.m
        gamma = self.sg ** 2
        R = gamma * torch.eye(n, n)
        Q = (-1 / gamma) * torch.eye(m, m)
        M = F.linear(self.X3, self.X3) + self.Y3 - self.Y3.T + self.epsilon * torch.eye(self.s)
        M_tilde = F.linear(torch.eye(self.s) - M,
                           torch.inverse(torch.eye(self.s) + M).T)
        Zeta = M_tilde[0:self.m, 0:self.n]
        self.D22 = gamma * Zeta
        R_capital = R - (1 / gamma) * F.linear(self.D22.T, self.D22.T)
        C2_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.C2)
        D21_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.D21) - self.D12.T
        vec_R = torch.cat([C2_capital.T, D21_capital.T, self.B2], 0)
        vec_Q = torch.cat([self.C2.T, self.D21.T, torch.zeros(n_xi, m)], 0)
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l) + torch.matmul(
            torch.matmul(vec_R, torch.inverse(R_capital)), vec_R.T) - torch.matmul(
            torch.matmul(vec_Q, Q), vec_Q.T)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l)
        vec[0] = 1
        epsilon = torch.zeros(self.l)
        v = F.linear(xi, self.C1[0, :]) + F.linear(w,
                                                   self.D12[0, :])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon,
                                                       self.D11[i, :]) + F.linear(w, self.D12[i, :])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon,
                                                self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + \
            F.linear(w, self.D22)  # + self.bu
        return u, xi_


class RENR2(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.m = m  # nel paper p
        self.s = np.max((n, m))  # s nel paper, dimensione di X3 Y3

        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l) * std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi) * std))  # Y1 nel paper
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n) * std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi) * std))
        self.D21 = nn.Parameter((torch.randn(m, l) * std))
        self.X3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.Y3 = nn.Parameter(torch.randn(self.s, self.s) * std)
        self.gamma = 0.7

        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n) * std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.Lq = torch.zeros(m, m)
        self.Lr = torch.zeros(n, n)
        self.D22 = torch.zeros(m, n)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        n = self.n
        m = self.m
        gamma = self.gamma
        R = gamma * torch.eye(n, n)
        Q = (-1 / gamma) * torch.eye(m, m)
        M = F.linear(self.X3, self.X3) + self.Y3 - self.Y3.T + self.epsilon * torch.eye(self.s)
        M_tilde = F.linear(torch.eye(self.s) - M,
                           torch.inverse(torch.eye(self.s) + M).T)
        Zeta = M_tilde[0:self.m, 0:self.n]
        self.D22 = gamma * Zeta
        R_capital = R - (1 / gamma) * F.linear(self.D22.T, self.D22.T)
        C2_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.C2)
        D21_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.D21) - self.D12.T
        vec_R = torch.cat([C2_capital.T, D21_capital.T, self.B2], 0)
        vec_Q = torch.cat([self.C2.T, self.D21.T, torch.zeros(n_xi, m)], 0)
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l) + torch.matmul(
            torch.matmul(vec_R, torch.inverse(R_capital)), vec_R.T) - torch.matmul(
            torch.matmul(vec_Q, Q), vec_Q.T)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l)
        vec[0] = 1
        epsilon = torch.zeros(self.l)
        v = F.linear(xi, self.C1[0, :]) + F.linear(w,
                                                   self.D12[0, :])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon,
                                                       self.D11[i, :]) + F.linear(w, self.D12[i, :])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon,
                                                self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + \
            F.linear(w, self.D22)  # + self.bu
        return u, xi_


class RENRG(nn.Module):
    def __init__(self, n, m, n_xi, l):
        super().__init__()
        self.gamma = torch.randn(1, 1, device=device, dtype=dtype)
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.m = m  # nel paper p
        self.s = np.max((n, m))  # s nel paper, dimensione di X3 Y3

        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1
        self.X = nn.Parameter((torch.randn(2 * n_xi + l, 2 * n_xi + l, device=device, dtype=dtype) * std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi, device=device, dtype=dtype) * std))  # Y1 nel paper
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n, device=device, dtype=dtype) * std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi, device=device, dtype=dtype) * std))
        self.D21 = nn.Parameter((torch.randn(m, l, device=device, dtype=dtype) * std))
        self.X3 = nn.Parameter(torch.randn(self.s, self.s, device=device, dtype=dtype) * std)
        self.Y3 = nn.Parameter(torch.randn(self.s, self.s, device=device, dtype=dtype) * std)

        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n, device=device, dtype=dtype) * std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi, device=device, dtype=dtype)
        self.B1 = torch.zeros(n_xi, l, device=device, dtype=dtype)
        self.E = torch.zeros(n_xi, n_xi, device=device, dtype=dtype)
        self.Lambda = torch.ones(l, device=device, dtype=dtype)
        self.C1 = torch.zeros(l, n_xi, device=device, dtype=dtype)
        self.D11 = torch.zeros(l, l, device=device, dtype=dtype)
        self.Lq = torch.zeros(m, m, device=device, dtype=dtype)
        self.Lr = torch.zeros(n, n, device=device, dtype=dtype)
        self.D22 = torch.zeros(m, n, device=device, dtype=dtype)
        self.set_model_param()

    def set_model_param(self):
        gamma = self.gamma
        n_xi = self.n_xi
        l = self.l
        n = self.n
        m = self.m
        R = gamma * torch.eye(n, n, device=device, dtype=dtype)
        Q = (-1 / gamma) * torch.eye(m, m, device=device, dtype=dtype)
        M = F.linear(self.X3, self.X3) + self.Y3 - self.Y3.T + self.epsilon * torch.eye(self.s, device=device,
                                                                                        dtype=dtype)
        M_tilde = F.linear(torch.eye(self.s, device=device, dtype=dtype) - M,
                           torch.inverse(torch.eye(self.s, device=device, dtype=dtype) + M).T)
        Zeta = M_tilde[0:self.m, 0:self.n]
        self.D22 = gamma * Zeta
        R_capital = R - (1 / gamma) * F.linear(self.D22.T, self.D22.T)
        C2_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.C2)
        D21_capital = torch.matmul(torch.matmul(self.D22.T, Q), self.D21) - self.D12.T
        vec_R = torch.cat([C2_capital.T, D21_capital.T, self.B2], 0)
        vec_Q = torch.cat([self.C2.T, self.D21.T, torch.zeros(n_xi, m, device=device, dtype=dtype)], 0)
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n_xi + l, device=device,
                                                                      dtype=dtype) + torch.matmul(
            torch.matmul(vec_R, torch.inverse(R_capital)), vec_R.T) - torch.matmul(
            torch.matmul(vec_Q, Q), vec_Q.T)
        h1, h2, h3 = torch.split(H, (n_xi, l, n_xi), dim=0)
        H11, H12, H13 = torch.split(h1, (n_xi, l, n_xi), dim=1)
        H21, H22, _ = torch.split(h2, (n_xi, l, n_xi), dim=1)
        H31, H32, H33 = torch.split(h3, (n_xi, l, n_xi), dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l, device=device, dtype=dtype)
        vec[0] = 1
        epsilon = torch.zeros(self.l, device=device, dtype=dtype)
        v = F.linear(xi, self.C1[0, :]) + F.linear(w,
                                                   self.D12[0, :])  # + self.bv[0]
        epsilon = epsilon + vec * torch.relu(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l, device=device, dtype=dtype)
            vec[i] = 1
            v = F.linear(xi, self.C1[i, :]) + F.linear(epsilon,
                                                       self.D11[i, :]) + F.linear(w, self.D12[i, :])  # self.bv[i]
            epsilon = epsilon + vec * torch.relu(v / self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon,
                                                self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + \
            F.linear(w, self.D22)  # + self.bu
        return u, xi_


class doubleRENg(nn.Module):
    def __init__(self, n, p, n_xi, n_xi2, l, l2):
        super().__init__()
        self.p = p
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.n_xi2 = n_xi2  # nel paper n
        self.l = l  # nel paper q
        self.l2 = l2  # nel paper q
        self.r1 = RENRG(self.n, self.p, self.n_xi, self.l)
        self.r2 = RENRG(self.p, self.n, self.n_xi2, self.l2)

        self.x1 = nn.Parameter(torch.randn(1, 1, device=device, dtype=dtype))
        self.x2 = nn.Parameter(torch.randn(1, 1, device=device, dtype=dtype))
        self.y1 = nn.Parameter(torch.randn(1, 1, device=device, dtype=dtype))
        self.y2 = nn.Parameter(torch.randn(1, 1, device=device, dtype=dtype))

        self.set_model_param()

    def set_model_param(self):
        x1 = self.x1 ** 2
        x2 = self.x2 ** 2
        y1 = self.y1 ** 2
        y2 = self.y2 ** 2
        gamma1 = torch.sqrt(y1 / (x1 + y2))
        gamma2 = torch.sqrt(y2 / (x2 + y1))
        self.r1.gamma = gamma1
        self.r2.gamma = gamma2

        self.r1.set_model_param()
        self.r2.set_model_param()

    def forward(self, t, u, xi, xi2):
        u1, xi = self.r1(t, u, xi)
        u2, xi2 = self.r2(t, u1, xi2)
        return u1, u2, xi, xi2


class multiRENL2(nn.Module):
    def __init__(self, N, M, n, p, n_xi, l):
        super().__init__()
        self.p = p
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.r = []
        self.M = M
        self.N = N
        for j in range(N):
            self.r.append(RENRG(self.n, self.p, self.n_xi[j], self.l[j]))

        self.pr = nn.Parameter(torch.randn(N, device=device, dtype=dtype))
        self.y = nn.Parameter(torch.randn(N, device=device, dtype=dtype))

        self.set_model_param()

    def set_model_param(self):
        M = self.M
        pr = self.pr
        y = self.y
        N = self.N
        pr2 = torch.exp(pr)
        y2 = torch.exp(y)
        PR = torch.diag(pr2)
        A = torch.matmul(torch.matmul(torch.transpose(M, 0, 1), PR), M)
        lmax = torch.trace(A)
        pv = lmax + y2
        P = torch.diag(pv)
        R = torch.matmul(torch.inverse(P), PR)
        gamma = R
        gamma = torch.sqrt(torch.diagonal(gamma, 0))

        for j in range(N):
            self.r[j].gamma = gamma[j]
            self.r[j].set_model_param()

    def forward(self, t, ym, v, xim):
        M = self.M
        N = self.N
        y = np.zeros(N)
        e = np.zeros(N)
        xi = []
        e[1] = 1.0
        for j in range(N):
            utemp = torch.matmul(M[j, :], ym)
            u = utemp.unsqueeze(dim=0)
            y[j], xi[j] = self.r[j](t, u + e * v, xim[j])
        return y, xi


class ThreeRENL2GershgorinShur(nn.Module):
    def __init__(self, N, M, n, p, n_xi, l):
        super().__init__()
        self.p = p
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.M = M
        self.N = N
        self.r = nn.ModuleList([RENRG(self.n, self.p, self.n_xi[j], self.l[j]) for j in range(N)])
        self.y = nn.Parameter(torch.randn(N, device=device, dtype=dtype))
        self.z = nn.Parameter(torch.randn(N, device=device, dtype=dtype))

        self.set_model_param()

    def set_model_param(self):
        M = self.M
        y = self.y
        z = self.z
        N = self.N
        y = torch.abs(y)
        z = torch.abs(z)
        pv = torch.sum(torch.abs(M), 0) + y
        x = torch.sum(torch.abs(M), 1) + z
        gamma = torch.sqrt(1 / (x * pv))

        for j in range(N):
            self.r[j].gamma = gamma[j]
            self.r[j].set_model_param()

    def forward(self, t, ym, v, xim):
        N = self.N
        y = torch.zeros(N)
        xi = []
        y[0], xitemp = self.r[0](t, (- ym + v).unsqueeze(0), xim[0])
        xi.append(xitemp)
        for j in range(1, N):
            y[j], xitemp2 = self.r[j](t, y[j - 1].unsqueeze(0).clone(), xim[j])
            xi.append(xitemp2)
        return y, xi


class ThreeRENL2GershgorinShurMIMO(nn.Module):
    def __init__(self, N, M, n, p, n_xi, l):
        super().__init__()
        self.p = p
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.M = M
        self.N = N
        self.r = nn.ModuleList([RENRG(self.n[j], self.p[j], self.n_xi[j], self.l[j]) for j in range(N)])
        self.y = nn.Parameter(torch.randn(N, device=device, dtype=dtype))
        self.z = nn.Parameter(torch.randn(N, device=device, dtype=dtype))

        self.set_model_param()

    def set_model_param(self):
        M = self.M
        y = self.y
        z = self.z
        N = self.N
        y = torch.abs(y)
        z = torch.abs(z)
        pv = torch.sum(torch.abs(M), 0) + y
        x = torch.sum(torch.abs(M), 1) + z
        gamma = torch.sqrt(1 / (x * pv))

        for j in range(N):
            self.r[j].gamma = gamma[j]
            self.r[j].set_model_param()

    def forward(self, t, ym, v, xim):
        y = torch.zeros(4)
        xi = []
        y[range(2)], xitemp = self.r[0](t, (- ym + v).unsqueeze(0), xim[0])
        xi.append(xitemp)
        y[2], xitemp2 = self.r[1](t, y[range(2)].unsqueeze(0).clone(), xim[1])
        xi.append(xitemp2)
        y[3], xitemp2 = self.r[2](t, y[2].unsqueeze(0).clone(), xim[2])
        xi.append(xitemp2)
        return y, xi


class pizzicottina(nn.Module):
    def __init__(self, N, M, n, p, n_xi, l):
        super().__init__()
        self.p = p
        self.n = n  # nel paper m
        self.n_xi = n_xi  # nel paper n
        self.l = l  # nel paper q
        self.M = M
        self.N = N
        self.r = nn.ModuleList([RENRG(self.n[j], self.p[j], self.n_xi[j], self.l[j]) for j in range(N)])
        self.y = nn.Parameter(torch.randn(N, device=device, dtype=dtype))
        self.z = nn.Parameter(torch.randn(N, device=device, dtype=dtype))

        self.set_model_param()

    def set_model_param(self):
        M = self.M
        y = self.y
        z = self.z
        N = self.N
        y = torch.abs(y)
        z = torch.abs(z)
        startu = 0
        stopu = 0
        starty = 0
        stopy = 0
        for j, l in enumerate(self.r):
            wideu = self.r[j].n
            widey = self.r[j].m
            startu = stopu
            stopu = stopu + wideu
            starty = stopy
            stopy = stopy + widey
            indexu = range(startu, stopu)
            indexy = range(starty, stopy)
            Mu = M[indexu, :]
            My = M[:, indexy]
            gamma = torch.sqrt((z[j] / (torch.max(torch.sum(torch.abs(Mu), 1)) * z[j] + 1))
                               / (1 + torch.max(torch.sum(torch.abs(My), 0)) + y[j]))
            self.r[j].gamma = gamma
            self.r[j].set_model_param()

    def forward(self, t, ym, d, xim):
        M = self.M
        u = torch.matmul(M, ym) + d
        y_list = []
        xi_list = []
        start = 0
        stop = 0
        startx = 0
        stopx = 0
        for j, l in enumerate(self.r):
            widex = self.r[j].n_xi
            startx = stopx
            stopx = stopx + widex
            wide = self.r[j].n
            start = stop
            stop = stop + wide
            index = range(start, stop)
            indexx = range(startx, stopx)
            yt, xitemp = self.r[j](t, u[index], xim[indexx])
            y_list.append(yt)
            xi_list.append(xitemp)
        y = torch.cat(y_list)
        xi = torch.cat(xi_list)
        return y, xi
