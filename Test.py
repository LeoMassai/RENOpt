import cvxpy as cp
import numpy as np

gg = 0.15
p1 = cp.Variable(1)
p2 = cp.Variable(1)
p3 = cp.Variable(1)
x1 = np.array([0, p2 / 2, p1 / 2])
x2 = np.array([p2 / 2, gg * p3, 0])
x3 = np.array([-p1 / 2, 0, -p3])

e1 = 0.68
e2 = 0.65

X = cp.bmat([[-e1 * p1, p2 / 2, -p1 / 2], [p2 / 2, -e2 * p2 + gg * p3, 0], [-p1 / 2, 0, -p3]])

constraints = [X << 0, p1 >= 0.002, p2 >= 0.002, p3 >= 0.002]

prob = cp.Problem(cp.Minimize(0),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(p1.value, p2.value, p3.value, e1*e2)
