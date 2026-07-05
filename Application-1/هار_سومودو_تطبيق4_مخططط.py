import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# الحل الدقيق
# -----------------------------
def exact_solution(x):
    return np.sin(x)

# -----------------------------
# الطرف الأيمن للمعادلة
# -----------------------------
def f_rhs(x):
    return np.cos(x) + np.sin(x) - 0.5*np.sin(x/2)

# -----------------------------
# مصفوفة Haar (تقريب بسيط)
# -----------------------------
def haar_matrix(x, N):
    H = np.zeros((len(x), N))
    h = (2*np.pi) / N
    for i, xi in enumerate(x):
        j = min(int(xi / h), N-1)
        H[i, j] = 1
    return H

# -----------------------------
# مشتقة كسرية (تقريب)
# -----------------------------
def fractional_derivative_matrix(N):
    D = np.zeros((N, N))
    for i in range(1, N):
        D[i, i] = 1
        D[i, i-1] = -1
    return D

# -----------------------------
# حل طريقة Haar–Sumudu
# -----------------------------
def solve_haar_sumudu(J, alpha):
    N = 2**(J+1)
    x = np.linspace(0, 2*np.pi, N)

    H = haar_matrix(x, N)
    H_half = haar_matrix(x/2, N)
    D = fractional_derivative_matrix(N)

    A = H @ D + H - 0.5 * H_half
    b = f_rhs(x)

    y = np.linalg.solve(A, b)
    return x, y

# -----------------------------
# الرسم
# -----------------------------
J = 8
alphas = [0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10,6))

# الحل الدقيق
x_exact = np.linspace(0, 2*np.pi, 1000)
plt.plot(x_exact, exact_solution(x_exact), label='Exact (α=1)', linewidth=3)

# الحلول التقريبية
for alpha in alphas:
    x, y = solve_haar_sumudu(J, alpha)
    plt.plot(x, y, label=f'α = {alpha}')

# تنسيق الرسم
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Haar–Sumudu Approximation (J=8)')
plt.legend()
plt.grid(True)

plt.show()
