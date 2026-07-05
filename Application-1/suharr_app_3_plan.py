import numpy as np
import matplotlib.pyplot as plt
from math import exp, gamma, isclose

# -------------------------------------------------
# إعدادات المسألة (معامل التأخير a = 4/5)
# -------------------------------------------------
ratio = 4.0 / 5.0      # a = 4/5
x_max = 10.0           # مجال الحل [0, 10]
J      = 8             # مستوى الدقة (N = 2^(J+1) * x_max)

# -------------------------------------------------
# الحل الدقيق للحالة α = 1
# -------------------------------------------------
def y_exact(x):
    return x * np.exp(-x)

# -------------------------------------------------
# معاملات مخطط L1 لمشتق كابوتو
# -------------------------------------------------
def L1_coeffs(N, alpha):
    b = np.empty(N + 1)
    b[0] = 0.0
    for k in range(1, N + 1):
        b[k] = k**(1 - alpha) - (k - 1)**(1 - alpha)
    return b

# -------------------------------------------------
# محلّل المعادلة الكسرية المؤجلة
# -------------------------------------------------
def solve_pantograph(alpha, J, x_max=10.0):
    """
    يحلّ:
        D^α y(x) - 0.5 D^α y(a x) = 0.1 y(a x) - y(x)
                                     + (0.32 x - 0.5) e^{-0.8 x} + e^{-x},
        مع y(0)=0، حيث a=4/5.
    """
    N = int(2 ** (J + 1) * x_max)  # عدد الفواصل
    h = x_max / N
    x_nodes = np.linspace(0.0, x_max, N + 1)

    y      = np.zeros(N + 1)
    dy_L1  = np.zeros(N + 1)       # تقريب D^α y
    b      = L1_coeffs(N, alpha)
    C      = h**(-alpha) / gamma(2 - alpha)

    for n in range(1, N + 1):
        xn = x_nodes[n]

        # موضع التأخير a x
        x_delay = ratio * xn
        k  = int(np.floor(x_delay / h))
        xk = k * h

        # استيفاء y و D^α y عند نقطة التأخير
        if isclose(x_delay, xk) or k == N:
            y_delay  = y[k]
            dy_delay = dy_L1[k]
        else:
            θ = (x_delay - xk) / h
            y_delay  = y[k]  + θ * (y[k + 1]  - y[k])
            dy_delay = dy_L1[k] + θ * (dy_L1[k + 1] - dy_L1[k])

        # الطرف الأيمن
        rhs = 0.1 * y_delay - y[n - 1] + (0.32 * xn - 0.5) * exp(-0.8 * xn) + exp(-xn)

        # تقريب المشتق الكسري عند xn
        diff = y[1:n][::-1] - y[: n - 1][::-1] if n > 1 else np.array([])
        prev_sum = np.dot(b[1:n], diff) if n > 1 else 0.0
        dy = 0.5 * dy_delay + rhs
        dy_L1[n] = dy

        # خطوة صريحة لتحديث y
        y[n] = y[n - 1] + h * dy   # جيّد لـ α≈1

    return x_nodes, y

# -------------------------------------------------
# حلّ واستخراج منحنيات لقيَم α مختلفة
# -------------------------------------------------
alphas = [0.6, 0.7, 0.8, 0.9]
solutions = {}

for a in alphas:
    xg, yg = solve_pantograph(a, J, x_max)
    solutions[a] = (xg, yg)

# -------------------------------------------------
# رسم المنحنيات
# -------------------------------------------------
x_exact = np.linspace(0.0, x_max, 2001)
plt.figure(figsize=(10, 5))
plt.plot(x_exact, y_exact(x_exact), label="Exact solution $y = xe^{-x}$")

for a in alphas:
    xg, yg = solutions[a]
    plt.plot(xg, yg, label=f"Approximate solution $\\alpha = {a}$, $J = 8$")

plt.title("Approximate Solutions on [0, 10] for $J = 8$")
plt.xlabel("$x$")
plt.ylabel("$y(x)$")
plt.legend()
plt.tight_layout()
plt.show()
