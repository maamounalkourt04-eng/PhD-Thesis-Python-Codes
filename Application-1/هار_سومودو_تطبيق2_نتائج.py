import numpy as np
import pandas as pd
from math import exp, isclose

# -------------------------------------------------
# حل معادلة التطبيق عند α = 1 (حالة هار–سومودو)
# -------------------------------------------------
def solve_u(J: int):
    """
    يحلّ معادلة:
        u'(x) = 2x - u(x) + u(x/2)
                + x(1+2x) ∫_0^x e^t (x-t) u(t) dt
                + ∫_0^{x/2} ½ x e^τ (½ x-τ) u(τ) dτ,
        مع الشرط الابتدائي u(0) = 1
    باستخدام مخطط مسير أمامي (Forward Marching)
    على شبكة منتظمة تحوي N = 2**(J+1) فواصل.
    """
    N = 2 ** (J + 1)          # عدد الفواصل
    h = 1.0 / N               # طول الفاصل
    x = np.linspace(0.0, 1.0, N + 1)
    u = np.zeros(N + 1)
    u[0] = 1.0                # u(0) = 1

    for i in range(1, N + 1):
        xi = x[i]

        # --------- التكامل الأول حتى x ---------
        t = x[: i + 1]
        integrand1 = np.exp(t) * (xi - t) * u[: i + 1]
        I1 = h * (0.5 * integrand1[0] +
                  integrand1[1:-1].sum() +
                  0.5 * integrand1[-1])

        # --------- التكامل الثاني حتى x/2 ---------
        x_half = xi / 2.0
        k = int(np.floor(x_half / h))       # آخر عقدة كاملة ≤ x/2
        if k == 0:
            I2 = 0.0
        else:
            t2 = x[: k + 1]
            integrand2 = 0.5 * xi * np.exp(t2) * (0.5 * xi - t2) * u[: k + 1]
            I2 = h * (0.5 * integrand2[0] +
                      integrand2[1:-1].sum() +
                      0.5 * integrand2[-1])

        # شريحة الذيل إذا كان x/2 ليس على عقدة
        xk = k * h
        if not isclose(x_half, xk) and k < i:
            u_k, u_k1 = u[k], u[k + 1]
            u_half = u_k + (u_k1 - u_k) * (x_half - xk) / h
            f_k = 0.5 * xi * np.exp(xk) * (0.5 * xi - xk) * u_k
            f_half = 0.5 * xi * np.exp(x_half) * (0.5 * xi - x_half) * u_half
            I2 += (x_half - xk) * 0.5 * (f_k + f_half)

        # --------- قيمة u(x/2) بالاستيفاء ---------
        if isclose(x_half, xk):
            u_half = u[k]
        else:
            u_half = u[k] + (u[k + 1] - u[k]) * (x_half - xk) / h

        # --------- معادلة خطية لـ u[i] ---------
        rhs = (2 * xi
               + u_half
               + xi * (1 + 2 * xi) * I1
               + I2)

        # تقريب u' ≈ (u_i - u_{i-1})/h + u_i على الجهة اليسرى
        u[i] = (rhs * h + u[i - 1]) / (1 + h)

    return x, u


def absolute_errors(J: int, sample_points):
    """يُرجع الأخطاء المطلقة عند نقاط sample_points لمستوى دقة J."""
    x_nodes, u_nodes = solve_u(J)
    h = x_nodes[1] - x_nodes[0]

    # استيفاء خطي بين العقد للحصول على u_h(x)
    def interp(xp):
        idx = int(np.floor(xp / h))
        if idx >= len(u_nodes) - 1:
            return u_nodes[-1]
        x0, x1 = x_nodes[idx], x_nodes[idx + 1]
        return u_nodes[idx] + (u_nodes[idx + 1] - u_nodes[idx]) * (xp - x0) / h

    exact = np.exp(sample_points ** 2)
    approx = np.array([interp(xp) for xp in sample_points])
    return np.abs(exact - approx)


# -------------------------------
# بناء جدول الأخطاء 
# -------------------------------
sample_x = np.arange(0.01, 0.101, 0.01)          # x_j من 0.01 إلى 0.10
rows = []

for xp in sample_x:
    row = [f"{xp:.2f}"]
    for J in (4, 6, 8):
        err = absolute_errors(J, np.array([xp]))[0]
        row.append(f"{err:.3e}")
    rows.append(row)

table = pd.DataFrame(
    rows,
    columns=[
        "x_j",
        "Absolute error (J=4)",
        "Absolute error (J=6)",
        "Absolute error (J=8)",
    ]
)

print("\nHaar–Sumudu absolute errors (α = 1)\n")
print(table.to_string(index=False))
