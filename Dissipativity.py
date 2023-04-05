import cvxpy as cp
import numpy as np

import sympy
from sympy import var
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt

var('e1 e2 g p1 p2 p3')

M = Matrix([[0, 0, -1], [1, 0, 0], [0, 1, 0]])

P = Matrix([[0, 1 / 2], [1 / 2, 0]])
S = Matrix([[0, 1 / 2], [1 / 2, -e1]])
SP2 = Matrix([[0, 1 / 2], [1 / 2, -e2]])
L = Matrix([[g, 0], [0, -1]])

supp = {
    "P": P,
    "S": S,
    "L": L
}

order = 'LLS'


def xp(order):
    x1 = Matrix.diag([p1 * supp[order[0]][0, 0], p2 * supp[order[1]][0, 0], p3 * supp[order[2]][0, 0]])
    x2 = Matrix.diag([p1 * supp[order[0]][0, 1], p2 * supp[order[1]][0, 1], p3 * supp[order[2]][0, 1]])
    x3 = Matrix.diag([p1 * supp[order[0]][1, 0], p2 * supp[order[1]][1, 0], p3 * supp[order[2]][1, 0]])
    x4 = Matrix.diag([p1 * supp[order[0]][1, 1], p2 * supp[order[1]][1, 1], p3 * supp[order[2]][1, 1]])
    Xp = Matrix([[x1, x2], [x3, x4]])
    mc = Matrix([[M], [Matrix.eye(3)]])
    Xp = mc.transpose() @ Xp @ mc
    return Xp


Xp = xp(order)

p1 = cp.Variable(1)
p2 = cp.Variable(1)
p3 = cp.Variable(1)
v1 = 5
v2 = 6
h = 2

Xp = Xp.subs([(e1, v1), (e2, v2), (g, h), (p1, p1), (p2, p2), (p3, p3)])

#
# Xp2=Xp.subs(p1,3)
#
#
# cp.bmat([[Xp[0, :]], [Xp[1, :]], [Xp[2, :]]])
