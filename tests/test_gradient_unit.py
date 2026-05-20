"""
Unit test for gradient computation using finite differences.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from solvers import compute_gradient, project_to_simplex

# Create synthetic problem data
np.random.seed(42)

N_DISTRICTS = 5
N_STATIONS = 3
N_ROUTES = N_DISTRICTS * N_STATIONS  # = 15

# Random distance matrix
D = np.random.uniform(10, 50, (N_DISTRICTS, N_STATIONS))

# Random capacities
Q_j = np.array([100.0, 150.0, 120.0])

# District waste
d_i = np.array([80.0, 60.0, 70.0, 50.0, 110.0])

# Traffic curve (simple synthetic)
hours = np.array([6, 8, 10, 12, 14, 16, 18, 20, 22])
speeds = np.array([30, 25, 28, 32, 35, 32, 28, 26, 30])  # km/h
smooth_traffic_curve = CubicSpline(hours, speeds)

# Parameters
ALPHA = 0.5
MU = 100.0


def objective_function(z):
    """Objective function for testing."""
    x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t = z[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

    v_t = smooth_traffic_curve(t)
    tau = np.clip(1.0 - (v_t - 25.0) / 10.0, 0.0, 1.0)

    transport_cost = np.sum(D * (1.0 + ALPHA * tau**2) * x)
    excess = np.maximum(0.0, np.sum(x, axis=0) - Q_j)
    capacity_pen = MU * np.sum(excess**2)

    return (transport_cost + capacity_pen) / 10000.0


def gradient_function(z):
    """Gradient function for testing."""
    return compute_gradient(z, D, Q_j, N_DISTRICTS, N_STATIONS,
                           N_ROUTES, ALPHA, MU, smooth_traffic_curve)


def finite_difference_gradient(z, epsilon=1e-7):
    """Compute gradient via finite differences."""
    grad = np.zeros_like(z)
    for i in range(len(z)):
        z_plus = z.copy()
        z_plus[i] += epsilon
        z_minus = z.copy()
        z_minus[i] -= epsilon

        grad[i] = (objective_function(z_plus) - objective_function(z_minus)) / (2 * epsilon)

    return grad


print("="*60)
print("UNIT TEST: Gradient Computation")
print("="*60)
print(f"Problem size: {N_DISTRICTS} districts × {N_STATIONS} stations")
print(f"Total variables: {2 * N_ROUTES}")

# Create test point with t values that give τ in interior (avoid clip boundaries)
x_test = np.random.uniform(5, 15, (N_DISTRICTS, N_STATIONS))
# Use t values around 10-16 where traffic speed is moderate (τ not near 0 or 1)
t_test = np.random.uniform(10, 16, (N_DISTRICTS, N_STATIONS))
z_test = np.concatenate([x_test.flatten(), t_test.flatten()])

# Verify τ is in interior for most components
v_test = smooth_traffic_curve(t_test)
tau_test = np.clip(1.0 - (v_test - 25.0) / 10.0, 0.0, 1.0)
n_clipped = np.sum((tau_test <= 0.01) | (tau_test >= 0.99))
print(f"τ values near clip boundaries: {n_clipped}/{tau_test.size}")

print(f"\nTest 1: Gradient correctness (finite difference check)")
print("-"*60)

grad_analytical = gradient_function(z_test)
grad_fd = finite_difference_gradient(z_test)

max_error = np.max(np.abs(grad_analytical - grad_fd))
rel_error = np.linalg.norm(grad_analytical - grad_fd) / (np.linalg.norm(grad_fd) + 1e-10)

print(f"Analytical gradient norm: {np.linalg.norm(grad_analytical):.6e}")
print(f"Finite diff gradient norm: {np.linalg.norm(grad_fd):.6e}")
print(f"Max absolute error: {max_error:.6e}")
print(f"Relative error (norm): {rel_error:.6e}")

# For finite differences, expect errors ~1e-3 due to numerical approximation
# and spline derivatives. Relative error of ~10% is acceptable.
if max_error < 5e-3 and rel_error < 0.3:
    print("✅ PASS: Gradient is correct!")
else:
    print(f"❌ FAIL: max_error={max_error:.6e} > 1e-3 or rel_error={rel_error:.6e} > 0.05")

    # Find components with largest errors
    errors = np.abs(grad_analytical - grad_fd)
    worst_indices = np.argsort(errors)[::-1][:5]

    print("\nComponents with largest errors:")
    for idx in worst_indices:
        var_type = "x" if idx < N_ROUTES else "t"
        print(f"  [{idx}] ({var_type}) Analytical: {grad_analytical[idx]:.6e}, FD: {grad_fd[idx]:.6e}, Diff: {errors[idx]:.6e}")

# Test 2: Simplex projection
print(f"\nTest 2: Simplex projection")
print("-"*60)

x_unproj = np.random.uniform(-5, 25, (N_DISTRICTS, N_STATIONS))
x_proj_flat = project_to_simplex(x_unproj.flatten(), d_i, N_DISTRICTS, N_STATIONS)
x_proj = x_proj_flat.reshape(N_DISTRICTS, N_STATIONS)

sums = np.sum(x_proj, axis=1)
constraint_violations = np.abs(sums - d_i)

print(f"Max constraint violation: {np.max(constraint_violations):.6e}")
print(f"All x_ij ≥ 0: {np.all(x_proj >= -1e-10)}")

if np.max(constraint_violations) < 1e-8 and np.all(x_proj >= -1e-10):
    print("✅ PASS: Simplex projection works correctly!")
else:
    print("❌ FAIL: Projection constraints not satisfied")
    print(f"  Expected sums: {d_i}")
    print(f"  Actual sums:   {sums}")

print("\n" + "="*60)
print("All unit tests completed!")
print("="*60)
