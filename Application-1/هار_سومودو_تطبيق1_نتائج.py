import numpy as np
import pandas as pd
from math import gamma, exp

alpha = 0.75

def exact_u(x):
    return x**3

def I1_exact(x):
    return exp(x)*(x**3 - 3*x**2 + 6*x - 6) + 6

def a4(x):
    if x == 0:
        return 0.0
    return exp(x) + 1 - 5*I1_exact(x)/(x**5)

def solve_problem(N):
    h = 1 / N
    x = np.linspace(0, 1, N + 1)
    u = np.zeros(N + 1)

    c = h**(-alpha) / gamma(2 - alpha)
    b = np.array([(k + 1)**(1 - alpha) - k**(1 - alpha) for k in range(N + 1)])

    for n in range(1, N + 1):
        xn = x[n]

        prev = -u[n-1]
        for k in range(1, n):
            prev += b[k] * (u[n-k] - u[n-k-1])

        I1_prev = h * sum(exp(x[i]) * u[i] for i in range(1, n))
        I2_prev = h * sum(x[i] * u[i] for i in range(1, n))

        F = 6 * xn**(3 - alpha) / gamma(4 - alpha)

        coeff = (
            c
            + xn**2 * (exp(xn) + 1) / 5
            - 0.5 * h * exp(xn)
            - 0.5 * h * a4(xn) * xn
        )

        rhs = F + I1_prev + a4(xn) * I2_prev - c * prev

        u[n] = rhs / coeff

    return x, u

test_points = np.arange(0.1, 1.01, 0.1)

rows = []

solutions = {}

for J, N in [(4, 32), (6, 128), (8, 512)]:
    x_grid, u_num = solve_problem(N)
    solutions[J] = (x_grid, u_num)

for xp in test_points:
    row = [round(xp, 1)]

    for J, N in [(4, 32), (6, 128), (8, 512)]:
        x_grid, u_num = solutions[J]
        index = int(round(xp * N))
        error = abs(exact_u(xp) - u_num[index])
        row.append(f"{error:.6e}")

    rows.append(row)

table = pd.DataFrame(
    rows,
    columns=[
        "x_j",
        "Absolute errors of J=4",
        "Absolute errors of J=6",
        "Absolute errors of J=8"
    ]
)

print(table.to_string(index=False))
