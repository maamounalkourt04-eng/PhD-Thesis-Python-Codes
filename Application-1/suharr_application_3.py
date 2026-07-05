import numpy as np
import pandas as pd
from math import exp, isclose

# -------------------------------------------------
# ثوابت المعادلة
# -------------------------------------------------
alpha = 1.0            # حالة المشتقّ الكلاسيكي
ratio = 4.0 / 5.0      # عامل التأخير  a = 4/5  ⇒  y(x/2) تُستبدل بـ y(ax)

# الحل الدقيق y = x e^{-x}
def exact_y(x):
    return x * np.exp(-x)

def exact_y_prime(x):
    return np.exp(-x) * (1 - x)

# -------------------------------------------------
# دالة حلّ المعادلة لمستوى دقّة معيّن J
# -------------------------------------------------
def solve_pantograph(J: int):
    """
    يحلّ المعادلة:
        y'(x) - 0.5 y'(a x) = 0.1 y(a x) - y(x)
                              + (0.32 x - 0.5) e^{-0.8 x} + e^{-x},
        مع y(0) = 0   و  a = 4/5،
    باستخدام مسير أمامي (Forward Marching) على شبكة منتظمة
    بعدد فواصل N = 2^(J+1).
    """
    N = 2 ** (J + 1)      # عدد الفواصل
    h = 1.0 / N           # طول الفاصل
    x = np.linspace(0.0, 1.0, N + 1)

    y = np.zeros(N + 1)
    y_prime = np.zeros(N + 1)

    # مُشتقّ ابتدائي عند x=0 (استعملنا القيمـة التحليلية للاستقرار)
    y_prime[0] = exact_y_prime(0)   # = 1

    # مسير أمامي عبر جميع العقد
    for i in range(1, N + 1):
        xi = x[i]

        # موضع التأخير  a x
        x_delay = ratio * xi
        k = int(np.floor(x_delay / h))      # أقرب عقدة يسار x_delay
        xk = k * h

        # استيفاء y و y' عند نقطة التأخير
        if isclose(x_delay, xk) or k == N:
            y_delay = y[k]
            yp_delay = y_prime[k]
        else:
            θ = (x_delay - xk) / h
            y_delay  = y[k]  + θ * (y[k + 1]  - y[k])
            yp_delay = y_prime[k] + θ * (y_prime[k + 1] - y_prime[k])

        # الطرف الأيمن للمعادلة
        rhs = (
            0.1 * y_delay
            - y[i - 1]                             # تقريب y(x) بقيمـة الخطوة السابقة (صريح)
            + (0.32 * xi - 0.5) * exp(-0.8 * xi)
            + exp(-xi)
        )

        # معادلة y'(x):  y' = 0.5 y'(a x) + rhs
        y_prime[i] = 0.5 * yp_delay + rhs

        # دمج أويلر لإيجاد y
        y[i] = y[i - 1] + h * y_prime[i]

    return x, y, y_prime

# -------------------------------------------------
# حساب الخطأ المطلق لعقد معيّنة
# -------------------------------------------------
def absolute_errors(J: int, sample_points):
    x_nodes, y_nodes, _ = solve_pantograph(J)
    h = x_nodes[1] - x_nodes[0]

    # استيفاء خطي للحصول على y_h عند أي x وسط العقد
    def interp(xp):
        idx = int(np.floor(xp / h))
        if idx >= len(y_nodes) - 1:
            return y_nodes[-1]
        x0, x1 = x_nodes[idx], x_nodes[idx + 1]
        return y_nodes[idx] + (y_nodes[idx + 1] - y_nodes[idx]) * (xp - x0) / h

    approx = np.array([interp(xp) for xp in sample_points])
    exact  = exact_y(sample_points)
    return np.abs(exact - approx)

# -------------------------------------------------
# توليد جدول الخطأ عند  J=4,6,8  ولـ x_j = 0.1..1.0
# -------------------------------------------------
sample_x = np.arange(0.1, 1.01, 0.1)
rows = []

for xp in sample_x:
    row = [f"{xp:.1f}"]
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
    ],
)

print("\nPantograph absolute errors (α = 1)\n")
print(table.to_string(index=False))
