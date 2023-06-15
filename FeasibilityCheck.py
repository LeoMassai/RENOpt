import cvxpy as cp
import numpy as np

np.zeros(4)

p1 = cp.Variable(1)
p2 = cp.Variable(1)
p3 = cp.Variable(1)
p4 = cp.Variable(1)
g1 = 24
pg2 = cp.Variable(1)
pg3 = cp.Variable(1)
pg4 = cp.Variable(1)

M = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

s = cp.bmat([[cp.diag(cp.hstack(
    [p1, p1, p1, p2, p3, p4])), M],
    [M, cp.diag(cp.hstack([cp.inv_pos(cp.multiply(cp.square(g1), p1)), cp.inv_pos(cp.multiply(cp.square(g1), p1)),
                           cp.inv_pos(cp.multiply(cp.square(g1), p1)),
                           pg2, pg3,
                           pg4]))]])

constraints = [s >> 0, p1 >= 0.001, p2 >= 0.001, p3 >= 0.001, p4 >= 0.001, pg2 >= 0.001, pg3 >= 0.001, pg4 >= 0.001]

prob = cp.Problem(cp.Minimize(0),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
