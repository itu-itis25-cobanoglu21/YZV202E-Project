"""
Compare all three solvers: Gradient Descent, SLSQP, and Newton (reduced).
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from solvers import gradient_descent_armijo, slsqp_solver, newton_reduced_problem, compute_gradient

# Load data
df_ilce_ref = pd.read_excel("data/ilce_koordinat.xlsx")
df_trafik = pd.read_excel("data/trafik.xlsx")

df_ilceler_raw = pd.read_excel("data/ilceler.xlsx")
df_ilceler_raw.columns = df_ilceler_raw.columns.astype(str).str.strip()
mevcut_yillar = [y for y in ['2021','2022','2023','2024','2025']
                 if y in df_ilceler_raw.columns]

df_ilceler_dinamik = pd.DataFrame({
    'Ilce_Adi': df_ilceler_raw['İlçe'].astype(str).str.strip().str.title(),
    'Yillik_Tonaj': df_ilceler_raw[mevcut_yillar].mean(axis=1)
})
df_ilceler = pd.merge(df_ilce_ref, df_ilceler_dinamik, on='Ilce_Adi', how='left')
df_ilceler['Yillik_Tonaj'] = df_ilceler['Yillik_Tonaj'].fillna(
    df_ilceler['Yillik_Tonaj'].mean())

df_istasyonlar_raw = pd.read_excel("data/istasyonlar.xlsx")
df_istasyonlar = pd.DataFrame({
    'Istasyon_Adi': df_istasyonlar_raw['AKTARMA İSTASYONLARI'].astype(str).str.strip(),
    'Enlem': df_istasyonlar_raw['LATITUDE'],
    'Boylam': df_istasyonlar_raw['LONGITUDE'],
    'Alan_m2': df_istasyonlar_raw['YÜZÖLÇÜMÜ (m2)']
})

Q_TOTAL = 12097.8
df_istasyonlar['Gunluk_Kapasite'] = (
    df_istasyonlar['Alan_m2'] / df_istasyonlar['Alan_m2'].sum()) * Q_TOTAL

asian_station_names = [
    "Küçük Bakkalköy Katı Atık Aktarma İstasyonu",
    "Hekimbaşı Katı Atık Aktama İstasyonu",
    "Aydınlı Katı Atık Aktarma İstasyonu",
    "Şile Katı Atık Aktarma İstasyonu"
]
df_istasyonlar['Yaka'] = np.where(
    df_istasyonlar['Istasyon_Adi'].isin(asian_station_names), 'Asya', 'Avrupa')

d_i_raw = (df_ilceler['Yillik_Tonaj'] / 365.0).to_numpy()
d_i_normalized = d_i_raw * (Q_TOTAL / d_i_raw.sum())
df_ilceler['Gunluk_Tonaj'] = d_i_normalized

# Setup problem
N_DISTRICTS = len(df_ilceler)
N_STATIONS = len(df_istasyonlar)
N_ROUTES = N_DISTRICTS * N_STATIONS

districts_list = df_ilceler['Ilce_Adi'].tolist()
d_i = df_ilceler['Gunluk_Tonaj'].to_numpy()
asian_districts = set(df_ilceler[df_ilceler['Yaka'] == 'Asya']['Ilce_Adi'])

station_names = df_istasyonlar['Istasyon_Adi'].tolist()
Q_j = df_istasyonlar['Gunluk_Kapasite'].to_numpy()
asian_stat_set = set(df_istasyonlar[df_istasyonlar['Yaka'] == 'Asya']['Istasyon_Adi'])

D = np.load("data/distance_matrix.npy")
smooth_traffic_curve = CubicSpline(df_trafik["Saat"], df_trafik["Hiz_kmh"])

# Build crossing mask
CROSSING_MASK = np.zeros((N_DISTRICTS, N_STATIONS), dtype=bool)
for i, d_name in enumerate(districts_list):
    for j, s_name in enumerate(station_names):
        if (d_name in asian_districts) != (s_name in asian_stat_set):
            CROSSING_MASK[i, j] = True

# Parameters
ALPHA = 0.5
MU = 500.0
CROSSING_PENALTY = 100.0

# Build penalized distance matrix
D_penalized = D.copy()
D_penalized[CROSSING_MASK] *= CROSSING_PENALTY


def objective_function(z):
    x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t = z[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

    v_t = smooth_traffic_curve(t)
    tau = np.clip(1.0 - (v_t - 27.0) / 9.0, 0.0, 1.0)

    transport_cost = np.sum(D_penalized * (1.0 + ALPHA * tau**2) * x)
    excess = np.maximum(0.0, np.sum(x, axis=0) - Q_j)
    capacity_pen = MU * np.sum(excess**2)

    return (transport_cost + capacity_pen) / 10000.0


def gradient_function(z):
    return compute_gradient(z, D_penalized, Q_j, N_DISTRICTS, N_STATIONS,
                           N_ROUTES, ALPHA, MU, smooth_traffic_curve)


# Initialize
x0_matrix = np.zeros((N_DISTRICTS, N_STATIONS))
for i, d_name in enumerate(districts_list):
    costs = D_penalized[i].copy()
    x0_matrix[i, int(np.argmin(costs))] = d_i[i]

z0 = np.concatenate([x0_matrix.flatten(), np.full(N_ROUTES, 14.0)])

print("="*80)
print("THREE SOLVER COMPARISON")
print("="*80)
print(f"Problem: {N_DISTRICTS} districts × {N_STATIONS} stations = {N_ROUTES} routes")
print(f"Total variables: {2 * N_ROUTES} = 702")
print(f"Initial objective: {objective_function(z0):.2f}")
print("="*80)

# Solver 1: Gradient Descent
print("\n" + "="*80)
print("1. GRADIENT DESCENT (Linear Convergence)")
print("="*80)
result_gd = gradient_descent_armijo(
    z0=z0.copy(),
    objective_fn=objective_function,
    gradient_fn=gradient_function,
    d_i=d_i,
    N_DISTRICTS=N_DISTRICTS,
    N_STATIONS=N_STATIONS,
    N_ROUTES=N_ROUTES,
    max_iter=100,
    tol=1e-6,
    verbose=True
)

# Solver 2: SLSQP
print("\n" + "="*80)
print("2. SLSQP (Superlinear Convergence)")
print("="*80)
result_slsqp = slsqp_solver(
    z0=z0.copy(),
    objective_fn=objective_function,
    gradient_fn=gradient_function,
    d_i=d_i,
    N_DISTRICTS=N_DISTRICTS,
    N_STATIONS=N_STATIONS,
    N_ROUTES=N_ROUTES,
    max_iter=100,
    tol=1e-6,
    verbose=True
)

# Solver 3: Newton on reduced problem (use SLSQP result as starting point)
print("\n" + "="*80)
print("3. NEWTON'S METHOD (Quadratic Convergence on Reduced Problem)")
print("="*80)
result_newton = newton_reduced_problem(
    z_init=result_slsqp['z_opt'],  # Start from SLSQP solution
    objective_fn=objective_function,
    gradient_fn=gradient_function,
    d_i=d_i,
    N_DISTRICTS=N_DISTRICTS,
    N_STATIONS=N_STATIONS,
    N_ROUTES=N_ROUTES,
    epsilon=5.0,
    max_iter=30,
    tol=1e-6,
    verbose=True
)

# Results table
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"{'Metric':<35} {'GD':<15} {'SLSQP':<15} {'Newton':<15}")
print("-"*80)
print(f"{'Final Objective:':<35} {result_gd['f_opt']:<15.2f} {result_slsqp['f_opt']:<15.2f} {result_newton['f_opt']:<15.2f}")
print(f"{'Iterations:':<35} {result_gd['iterations']:<15} {result_slsqp['iterations']:<15} {result_newton['iterations']:<15}")
print(f"{'Converged:':<35} {str(result_gd['converged']):<15} {str(result_slsqp['converged']):<15} {str(result_newton['converged']):<15}")
print(f"{'Final ||∇f||:':<35} {result_gd['grad_norms'][-1]:<15.2e} {result_slsqp['grad_norms'][-1]:<15.2e} {result_newton['grad_norms'][-1]:<15.2e}")
if 'n_active' in result_newton:
    print(f"{'Problem size (variables):':<35} {'702':<15} {'702':<15} {str(2*result_newton['n_active']):<15}")

# Plot convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Objective value
ax1.semilogy(result_gd['history'], 'b-', label='GD (linear)', linewidth=2, alpha=0.8)
ax1.semilogy(result_slsqp['history'], 'r-', label='SLSQP (superlinear)', linewidth=2, alpha=0.8)

# For Newton, show as refinement starting from SLSQP's final value
newton_history_full = [result_slsqp['history'][-1]] + result_newton['history']
newton_iters = np.arange(result_slsqp['iterations'], result_slsqp['iterations'] + len(newton_history_full))
ax1.semilogy(newton_iters, newton_history_full, 'g-', label='Newton (quadratic)', linewidth=2, marker='o', markersize=4)

ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Objective Value f(x,t)', fontsize=11)
ax1.set_title('Convergence Comparison: Objective Value', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Gradient norm
ax2.semilogy(result_gd['grad_norms'], 'b-', label='GD (linear)', linewidth=2, alpha=0.8)
ax2.semilogy(result_slsqp['grad_norms'], 'r-', label='SLSQP (superlinear)', linewidth=2, alpha=0.8)
newton_grad_full = [result_slsqp['grad_norms'][-1]] + result_newton['grad_norms']
ax2.semilogy(newton_iters, newton_grad_full, 'g-', label='Newton (quadratic)', linewidth=2, marker='o', markersize=4)

ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Gradient Norm ||∇f||', fontsize=11)
ax2.set_title('Convergence Comparison: Gradient Norm', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/all_solvers_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Convergence plot saved to outputs/all_solvers_comparison.png")

print("\n" + "="*80)
print("All tests completed!")
print("="*80)
