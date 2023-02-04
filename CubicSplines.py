import numpy as np
import matplotlib.pyplot as plt
import project2
x = [i[0] for i in project2.maxs]
y = [i[1] for i in project2.maxs]

""" cubic spline """
n = len(x)
B = np.zeros((1, 4 * (n - 1)))
plt.scatter(x, y)
B[0, 0:2 * (n - 1)] = np.array([[y[i], y[i + 1]] for i in range(len(y)) if i + 1 < len(y)]).flatten().reshape(1, 2 * (
        n - 1)).squeeze()
A = np.zeros((4 * (n - 1), 4 * (n - 1)))
for i in range(n - 1):
    coeff = np.array(
        [x[i] ** 3, x[i] ** 2, x[i] ** 1, x[i] ** 0, x[i + 1] ** 3, x[i + 1] ** 2, x[i + 1] ** 1,
         x[i + 1] ** 0]).reshape((2, 4))
    A[(2 * i): (2 * i + 2), (4 * i):(4 * i + 4)] = coeff
    if i < n - 2:
        firstDerivativeCoeff = np.array(
            [3 * x[i + 1] ** 2, 2 * x[i + 1] ** 1, x[i + 1] ** 0, 0, -3 * x[i + 1] ** 2, -2 * x[i + 1] ** 1,
             -x[i + 1] ** 0, 0]).reshape(1, 8)
        A[2 * (n - 1) + i, (4 * i):(4 * i + 8)] = firstDerivativeCoeff
    if i < n - 2:
        secondDerivativeCoeff = np.array(
            [6 * x[i + 1] ** 1, 2 * x[i + 1] ** 0, 0, 0, -6 * x[i + 1] ** 1, -2 * x[i + 1] ** 0, 0, 0, ]).reshape(1, 8)
        A[2 * (n - 1) + (n - 2) + i, (4 * i):(4 * i + 8)] = secondDerivativeCoeff
A[-2, 0:2] = [6 * x[0], 2]
A[-1, -4:-2] = [6 * x[n - 1], 2]

coeffs = np.linalg.solve(A, B.T)
for i in range(n - 1):
    domain = np.linspace(x[i], x[i + 1], 20)
    cubicCoeff = coeffs[(0 + 4 * i):(4 + 4 * i)]
    f = cubicCoeff[0] * (domain ** 3) + cubicCoeff[1] * (domain ** 2) + cubicCoeff[2] * (domain ** 1) + cubicCoeff[
        3] * (domain ** 0)
    plt.plot(domain, f, color="black")
plt.show()
""" """
