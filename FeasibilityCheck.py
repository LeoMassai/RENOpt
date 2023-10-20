import cvxpy as cp
import numpy as np
from numpy import linalg

np.zeros(4)

p1 = cp.Variable(1)
p2 = cp.Variable(1)
p3 = cp.Variable(1)
p4 = cp.Variable(1)
g1 = 26
pg2 = cp.Variable(1)
pg3 = cp.Variable(1)
pg4 = cp.Variable(1)
Y = cp.Variable((12, 12))

M = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

Pny = cp.diag(cp.vstack(
    [p1, p1, p1, p2, p3, p4]))

PR = cp.diag(cp.hstack([cp.multiply(g1, p1), cp.multiply(g1, p1),
                        cp.multiply(g1, p1),
                        pg2, pg3,
                        pg4]))
X = Pny - M.T @ PR @ M

constraints = [X >> 0, p1 >= 0.001, p2 >= 0.001, p3 >= 0.001, p4 >= 0.001, pg2 >= 0.001,
               pg3 >= 0.001, pg4 >= 0.001]

prob = cp.Problem(cp.Minimize(0),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(p1.value, p2.value, p3.value, p4.value, pg2.value / p2.value, pg3.value / p3.value, pg4.value / p4.value)

np.linalg.eigvals(X.value)

XV = X.value
