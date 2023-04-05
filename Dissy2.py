import cvxpy as cp
import numpy as np

import sympy
from sympy import var
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt

e1 = 0.3
e2 = 0.4
g = 0.01
g2 = 0.03
p1 = cp.Variable(1)
p2 = cp.Variable(1)
p3 = cp.Variable(1)

M = cp.bmat([[0, 0, -1], [1, 0, 0], [0, 1, 0]])

P = cp.bmat([[0, 1 / 2], [1 / 2, 0]])
S = cp.bmat([[0, 1 / 2], [1 / 2, -e1]])
S2 = cp.bmat([[0, 1 / 2], [1 / 2, -e2]])
L = cp.bmat([[g, 0], [0, -1]])
L2 = cp.bmat([[g2, 0], [0, -1]])

supp = {
    "P": P,
    "S": S,
    "L": L,
    "S2": S2,
    "L2": L2
}

order = ['L2', 'S', 'L']


def xp(order):
    x1 = cp.diag(cp.hstack([p1 * supp[order[0]][0, 0], p2 * supp[order[1]][0, 0], p3 * supp[order[2]][0, 0]]))
    x2 = cp.diag(cp.hstack([p1 * supp[order[0]][0, 1], p2 * supp[order[1]][0, 1], p3 * supp[order[2]][0, 1]]))
    x3 = cp.diag(cp.hstack([p1 * supp[order[0]][1, 0], p2 * supp[order[1]][1, 0], p3 * supp[order[2]][1, 0]]))
    x4 = cp.diag(cp.hstack([p1 * supp[order[0]][1, 1], p2 * supp[order[1]][1, 1], p3 * supp[order[2]][1, 1]]))
    Xp = cp.bmat([[x1, x2], [x3, x4]])
    mc = cp.bmat([[M], [np.eye(3)]])
    Xp = mc.T @ Xp @ mc
    return Xp


Xp = xp(order)

constraints = [Xp << 0, p1 >= 0.001, p2 >= 0.001, p3 >= 0.001]

prob = cp.Problem(cp.Minimize(0),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(p1.value, p2.value, p3.value, (np.sqrt(g)) ** 3)
