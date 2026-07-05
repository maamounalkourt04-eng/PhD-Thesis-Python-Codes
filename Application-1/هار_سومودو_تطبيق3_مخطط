import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def f_rhs(x):
    return (0.32*x - 0.5)*np.exp(-0.8*x) + np.exp(-x)

def exact_solution(x):
    return x*np.exp(-x)

def caputo_matrix(points, alpha, N, L):
    h = L / N
    A = np.zeros((len(points), N))

    for r, x in enumerate(points):
        if x <= 0:
            continue

        mmax = min(int(np.ceil(x / h)), N)

        for m in range(1, mmax + 1):
            t0 = (m - 1) * h
            t1 = m * h

            if t0 >= x:
                break

            b = min(t1, x)

            coeff = ((x - t0)**(1-alpha) - (x - b)**(1-alpha)) / (h * gamma(2-alpha))

            A[r, m-1] += coeff

            if m-2 >= 0:
                A[r, m-2] -= coeff

    return A

def interpolation_matrix(points, N, L):
    h = L / N
    M = np.zeros((len(points), N))

    for r, x in enumerate(points):
        if x <= 0:
            continue

        if x >= L:
            M[r, N-1] = 1.0
            continue

        k = int(np.floor(x / h))
        theta = (x - k*h) / h

        if k == 0:
            M[r, 0] = theta
        else:
            M[r, k-1] = 1.0 - theta
            M[r, k] = theta

    return M

def solve_fractional_pantograph(alpha, J, L):
    N = 2**(J+1)
    q = 4/5

    x_nodes = np.linspace(L/N, L, N)
    q_nodes = q * x_nodes

    D_x = caputo_matrix(x_nodes, alpha, N, L)
    D_q = caputo_matrix(q_nodes, alpha, N, L)
    Y_q = interpolation_matrix(q_nodes, N, L)

    I = np.eye(N)

    A = D_x - 0.5*D_q - 0.1*Y_q + I
    b = f_rhs(x_nodes)

    y_nodes = np.linalg.solve(A, b)

    x_full = np.concatenate(([0.0], x_nodes))
    y_full = np.concatenate(([0.0], y_nodes))

    return x_full, y_full

J = 8
L = 10.0
alphas = [0.7, 0.8, 0.9]

x_plot = np.linspace(0, L, 2000)
y_exact = exact_solution(x_plot)

plt.figure(figsize=(9, 5.5))

plt.plot(x_plot, y_exact, linewidth=2.5,
         label=r'Exact solution $y=xe^{-x}$')

for alpha in alphas:
    x_alpha, y_alpha = solve_fractional_pantograph(alpha=alpha, J=J, L=L)
    y_plot = np.interp(x_plot, x_alpha, y_alpha)

    plt.plot(x_plot, y_plot, linewidth=2,
             label=fr'Approximate solution $\alpha={alpha}$, $J=8$')

plt.xlabel('x')
plt.ylabel('y(x)')
plt.title(r'Approximate Solutions on $[0,10]$ for $J=8$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
