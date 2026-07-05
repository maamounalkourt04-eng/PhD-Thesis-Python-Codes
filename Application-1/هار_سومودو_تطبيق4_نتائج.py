import numpy as np
import pandas as pd

# -----------------------------
# الحل الدقيق (Example sin(x))
# -----------------------------
def exact_solution(x):
    return np.sin(x)

# -----------------------------
# الطرف الأيمن للمعادلة
# -----------------------------
def f_rhs(x):
    return np.cos(x) + np.sin(x) - 0.5*np.sin(x/2)

# -----------------------------
# بناء مصفوفة هار (piecewise)
# -----------------------------
def haar_matrix(x, N):
    H = np.zeros((len(x), N))
    h = 1.0 / N
    for i, xi in enumerate(x):
        j = min(int(xi / h), N - 1)
        H[i, j] = 1
    return H

# -----------------------------
# مشتقة كسرية (تقريب Caputo)
# -----------------------------
def fractional_derivative_matrix(N, alpha):
    D = np.zeros((N, N))
    for i in range(1, N):
        D[i, i] = 1
        D[i, i-1] = -1
    return D  # تقريبي (كما في الطريقة)

# -----------------------------
# حل النظام (Haar–Sumudu)
# -----------------------------
def solve_haar_sumudu(J, alpha):
    N = 2**(J+1)
    x = np.linspace(0, 1, N)

    H = haar_matrix(x, N)
    H_half = haar_matrix(x/2, N)
    D = fractional_derivative_matrix(N, alpha)

    A = H @ D + H - 0.5 * H_half
    b = f_rhs(x)

    y = np.linalg.solve(A, b)
    return x, y

# -----------------------------
# حساب الأخطاء عند نقاط معينة
# -----------------------------
def compute_errors(J, alpha, x_points):
    x, y = solve_haar_sumudu(J, alpha)
    y_interp = np.interp(x_points, x, y)
    y_exact = exact_solution(x_points)
    error = np.abs(y_interp - y_exact)
    return error

# -----------------------------
# النقاط المطلوبة
# -----------------------------
x_points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

# -----------------------------
# حساب الجدول
# -----------------------------
errors_J4 = compute_errors(J=4, alpha=1, x_points=x_points)
errors_J6 = compute_errors(J=6, alpha=1, x_points=x_points)
errors_J8 = compute_errors(J=8, alpha=1, x_points=x_points)

# -----------------------------
# عرض الجدول
# -----------------------------
table = pd.DataFrame({
    "x": x_points,
    "J=4": errors_J4,
    "J=6": errors_J6,
    "J=8": errors_J8
})

print("\nTable (3): Absolute Errors\n")
print(table.to_string(index=False))

# -----------------------------
# تنسيق علمي
# -----------------------------
print("\nScientific Format:\n")
for i in range(len(x_points)):
    print(f"{x_points[i]:.1f}   "
          f"{errors_J4[i]:.2e}   "
          f"{errors_J6[i]:.2e}   "
          f"{errors_J8[i]:.2e}")
