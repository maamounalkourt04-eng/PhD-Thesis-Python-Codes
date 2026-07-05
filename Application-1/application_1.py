import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================
# Generate Sample Data (Replace with Actual Data)
# =============================================

# Spatial domain (x) and time domain (t)
x = np.linspace(0, 1, 30)  # 30 spatial points
t_uniform = np.arange(0, 1.01, 0.01)  # Uniform time steps
t_adaptive = np.sort(np.random.uniform(0, 1, 42))  # Adaptive time steps (random for illustration)

# Exact solution (example function)
def exact_solution(x, t):
    return (t**2.5) * np.sin(np.pi * x)  # t^(2+α) with α=1.5

# Numerical solutions (replace with actual computed values)
def uniform_numerical(x, t):
    return exact_solution(x, t) * (1 - 0.02*np.exp(-t))  # Example approximation

def adaptive_numerical(x, t):
    return exact_solution(x, t) * (1 - 0.005*np.exp(-2*t))  # Better approximation

# Create meshgrids
X_unif, T_unif = np.meshgrid(x, t_uniform)
X_adapt, T_adapt = np.meshgrid(x, t_adaptive)

# Compute solutions
U_exact_unif = exact_solution(X_unif, T_unif)
U_num_unif = uniform_numerical(X_unif, T_unif)

U_exact_adapt = exact_solution(X_adapt, T_adapt)
U_num_adapt = adaptive_numerical(X_adapt, T_adapt)

# =================================
# Create 3D Plots (Separate Figures)
# =================================

# Figure 1: Uniform Time-Stepping
fig1 = plt.figure(figsize=(12, 5))

# Subplot 1: Exact Solution
ax1 = fig1.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X_unif, T_unif, U_exact_unif, cmap='viridis')
ax1.set_title('Exact Solution\n(Reference Solution)', fontsize=12)
ax1.set_xlabel('Space (x)', fontsize=10)
ax1.set_ylabel('Time (t)', fontsize=10)
ax1.set_zlabel('u(x,t)', fontsize=10)
ax1.view_init(30, 45)  # Adjust viewing angle
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Subplot 2: Numerical Solution (Uniform)
ax2 = fig1.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X_unif, T_unif, U_num_unif, cmap='plasma')
ax2.set_title('Numerical Solution\n(Uniform Δt=0.01)', fontsize=12)
ax2.set_xlabel('Space (x)', fontsize=10)
ax2.set_ylabel('Time (t)', fontsize=10)
ax2.set_zlabel('u(x,t)', fontsize=10)
ax2.view_init(30, 45)  # Match viewing angle
fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()

# Figure 2: Adaptive Time-Stepping
fig2 = plt.figure(figsize=(12, 5))

# Subplot 1: Exact Solution
ax3 = fig2.add_subplot(121, projection='3d')
surf3 = ax3.plot_surface(X_adapt, T_adapt, U_exact_adapt, cmap='viridis')
ax3.set_title('Exact Solution\n(Reference Solution)', fontsize=12)
ax3.set_xlabel('Space (x)', fontsize=10)
ax3.set_ylabel('Time (t)', fontsize=10)
ax3.set_zlabel('u(x,t)', fontsize=10)
ax3.view_init(30, 45)
fig2.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

# Subplot 2: Numerical Solution (Adaptive)
ax4 = fig2.add_subplot(122, projection='3d')
surf4 = ax4.plot_surface(X_adapt, T_adapt, U_num_adapt, cmap='plasma')
ax4.set_title('Numerical Solution\n(Adaptive Time-Stepping)', fontsize=12)
ax4.set_xlabel('Space (x)', fontsize=10)
ax4.set_ylabel('Time (t)', fontsize=10)
ax4.set_zlabel('u(x,t)', fontsize=10)
ax4.view_init(30, 45)
fig2.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)

plt.tight_layout()

# Figure 3: Error Comparison
fig3 = plt.figure(figsize=(12, 5))

# Error for uniform stepping
error_unif = np.abs(U_exact_unif - U_num_unif)

# For adaptive stepping, we need to interpolate to common grid for comparison
from scipy.interpolate import griddata
points = np.column_stack((X_adapt.ravel(), T_adapt.ravel()))
error_adapt = np.abs(U_exact_adapt - U_num_adapt)
error_adapt_grid = griddata(points, error_adapt.ravel(), (X_unif, T_unif), method='cubic')

# Plot uniform error
ax5 = fig3.add_subplot(121, projection='3d')
surf5 = ax5.plot_surface(X_unif, T_unif, error_unif, cmap='hot')
ax5.set_title('Absolute Error\n(Uniform Time-Stepping)', fontsize=12)
ax5.set_xlabel('Space (x)', fontsize=10)
ax5.set_ylabel('Time (t)', fontsize=10)
ax5.set_zlabel('Error', fontsize=10)
ax5.view_init(30, 45)
fig3.colorbar(surf5, ax=ax5, shrink=0.5, aspect=5)

# Plot adaptive error
ax6 = fig3.add_subplot(122, projection='3d')
surf6 = ax6.plot_surface(X_unif, T_unif, error_adapt_grid, cmap='hot')
ax6.set_title('Absolute Error\n(Adaptive Time-Stepping)', fontsize=12)
ax6.set_xlabel('Space (x)', fontsize=10)
ax6.set_ylabel('Time (t)', fontsize=10)
ax6.set_zlabel('Error', fontsize=10)
ax6.view_init(30, 45)
fig3.colorbar(surf6, ax=ax6, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
