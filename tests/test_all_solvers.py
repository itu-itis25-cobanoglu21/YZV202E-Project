"""
Compare all four solvers: Gradient Descent, L-BFGS, SLSQP, and Newton (reduced).

Run from project root:
    python tests/test_all_solvers.py
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from solvers import (gradient_descent_armijo, lbfgs_solver,
                     slsqp_solver, newton_reduced_problem, compute_gradient)

# ── Data Loading ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data')

df_ilce_ref    = pd.read_excel(os.path.join(DATA, "ilce_koordinat.xlsx"))
df_trafik      = pd.read_excel(os.path.join(DATA, "trafik.xlsx"))
df_ilceler_raw = pd.read_excel(os.path.join(DATA, "ilceler.xlsx"))
df_ilceler_raw.columns = df_ilceler_raw.columns.astype(str).str.strip()
available_years = [y for y in ['2021','2022','2023','2024','2025']
                   if y in df_ilceler_raw.columns]

df_ilceler_dyn = pd.DataFrame({
    'Ilce_Adi':     df_ilceler_raw['İlçe'].astype(str).str.strip().str.title(),
    'Yillik_Tonaj': df_ilceler_raw[available_years].mean(axis=1)
})
df_ilceler = pd.merge(df_ilce_ref, df_ilceler_dyn, on='Ilce_Adi', how='left')
df_ilceler['Yillik_Tonaj'] = df_ilceler['Yillik_Tonaj'].fillna(
    df_ilceler['Yillik_Tonaj'].mean())

df_istasyonlar_raw = pd.read_excel(os.path.join(DATA, "istasyonlar.xlsx"))
df_istasyonlar = pd.DataFrame({
    'Istasyon_Adi': df_istasyonlar_raw['AKTARMA İSTASYONLARI'].astype(str).str.strip(),
    'Enlem':        df_istasyonlar_raw['LATITUDE'],
    'Boylam':       df_istasyonlar_raw['LONGITUDE'],
    'Alan_m2':      df_istasyonlar_raw['YÜZÖLÇÜMÜ (m2)']
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

d_i_raw        = (df_ilceler['Yillik_Tonaj'] / 365.0).to_numpy()
d_i_normalized = d_i_raw * (Q_TOTAL / d_i_raw.sum())
df_ilceler['Gunluk_Tonaj'] = d_i_normalized

# ── Problem Setup ─────────────────────────────────────────────────────────────
N_DISTRICTS = len(df_ilceler)
N_STATIONS  = len(df_istasyonlar)
N_ROUTES    = N_DISTRICTS * N_STATIONS

districts_list  = df_ilceler['Ilce_Adi'].tolist()
d_i             = df_ilceler['Gunluk_Tonaj'].to_numpy()
asian_districts = set(df_ilceler[df_ilceler['Yaka'] == 'Asya']['Ilce_Adi'])

station_names  = df_istasyonlar['Istasyon_Adi'].tolist()
Q_j            = df_istasyonlar['Gunluk_Kapasite'].to_numpy()
asian_stat_set = set(df_istasyonlar[df_istasyonlar['Yaka'] == 'Asya']['Istasyon_Adi'])

D = np.load(os.path.join(DATA, "distance_matrix.npy"))
smooth_traffic_curve = CubicSpline(df_trafik["Saat"], df_trafik["Hiz_kmh"])

CROSSING_MASK = np.zeros((N_DISTRICTS, N_STATIONS), dtype=bool)
for i, d_name in enumerate(districts_list):
    for j, s_name in enumerate(station_names):
        if (d_name in asian_districts) != (s_name in asian_stat_set):
            CROSSING_MASK[i, j] = True

ALPHA            = 0.5
MU               = 1500.0
CROSSING_PENALTY = 100.0

D_penalized = D.copy()
D_penalized[CROSSING_MASK] *= CROSSING_PENALTY


def objective_function(z):
    x   = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t   = z[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)
    v_t = smooth_traffic_curve(t)
    tau = np.clip(1.0 - (v_t - 27.0) / 9.0, 0.0, 1.0)
    transport_cost = np.sum(D_penalized * (1.0 + ALPHA * tau**2) * x)
    excess         = np.maximum(0.0, np.sum(x, axis=0) - Q_j)
    capacity_pen   = MU * np.sum(excess**2)
    return (transport_cost + capacity_pen) / 10000.0


def gradient_function(z):
    return compute_gradient(z, D_penalized, Q_j, N_DISTRICTS, N_STATIONS,
                            N_ROUTES, ALPHA, MU, smooth_traffic_curve)


# Initial point: each district to nearest same-side station
x0_matrix = np.zeros((N_DISTRICTS, N_STATIONS))
for i, d_name in enumerate(districts_list):
    costs = D_penalized[i].copy()
    x0_matrix[i, int(np.argmin(costs))] = d_i[i]
z0 = np.concatenate([x0_matrix.flatten(), np.full(N_ROUTES, 14.0)])

SEP = "=" * 72
print(SEP)
print("FOUR-SOLVER COMPARISON: GD | L-BFGS | SLSQP | Newton")
print(SEP)
print(f"Problem : {N_DISTRICTS} districts x {N_STATIONS} stations = {N_ROUTES} routes")
print(f"Variables: {2 * N_ROUTES}   |   MU={MU}   |   ALPHA={ALPHA}")
print(f"Initial objective: {objective_function(z0):.4f}")
print(SEP)

MAX_ITER = 100

# ── Solver 1: Gradient Descent ────────────────────────────────────────────────
print(f"\n{SEP}\n1. GRADIENT DESCENT + ARMIJO  (linear convergence)\n{SEP}")
result_gd = gradient_descent_armijo(
    z0=z0.copy(), objective_fn=objective_function, gradient_fn=gradient_function,
    d_i=d_i, N_DISTRICTS=N_DISTRICTS, N_STATIONS=N_STATIONS, N_ROUTES=N_ROUTES,
    max_iter=MAX_ITER, tol=1e-6, verbose=True
)

# ── Solver 2: L-BFGS ──────────────────────────────────────────────────────────
print(f"\n{SEP}\n2. L-BFGS-B  (superlinear, limited-memory quasi-Newton + projection)\n{SEP}")
result_lbfgs = lbfgs_solver(
    z0=z0.copy(), objective_fn=objective_function, gradient_fn=gradient_function,
    d_i=d_i, N_DISTRICTS=N_DISTRICTS, N_STATIONS=N_STATIONS, N_ROUTES=N_ROUTES,
    max_iter=MAX_ITER, tol=1e-6, verbose=True
)

# ── Solver 3: SLSQP ───────────────────────────────────────────────────────────
print(f"\n{SEP}\n3. SLSQP  (superlinear, native equality constraints via BFGS)\n{SEP}")
result_slsqp = slsqp_solver(
    z0=z0.copy(), objective_fn=objective_function, gradient_fn=gradient_function,
    d_i=d_i, N_DISTRICTS=N_DISTRICTS, N_STATIONS=N_STATIONS, N_ROUTES=N_ROUTES,
    max_iter=MAX_ITER, tol=1e-6, verbose=True
)

# ── Solver 4: Newton reduced (warm-started from best quasi-Newton result) ─────
best_z = (result_slsqp['z_opt'] if result_slsqp['f_opt'] <= result_lbfgs['f_opt']
          else result_lbfgs['z_opt'])
print(f"\n{SEP}\n4. NEWTON (reduced problem, quadratic convergence)\n"
      f"   Warm-start: {'SLSQP' if result_slsqp['f_opt'] <= result_lbfgs['f_opt'] else 'L-BFGS'}"
      f"   f={min(result_slsqp['f_opt'], result_lbfgs['f_opt']):.4f}\n{SEP}")
result_newton = newton_reduced_problem(
    z_init=best_z, objective_fn=objective_function, gradient_fn=gradient_function,
    d_i=d_i, N_DISTRICTS=N_DISTRICTS, N_STATIONS=N_STATIONS, N_ROUTES=N_ROUTES,
    epsilon=5.0, max_iter=30, tol=1e-6, verbose=True
)

# ── Results Table ─────────────────────────────────────────────────────────────
print(f"\n{SEP}\nRESULTS SUMMARY\n{SEP}")
cols = ['GD', 'L-BFGS', 'SLSQP', 'Newton']
res  = [result_gd, result_lbfgs, result_slsqp, result_newton]
print(f"{'Metric':<32} " + " ".join(f"{c:<13}" for c in cols))
print("-" * 72)
print(f"{'Final objective f(x,t):':<32} " +
      " ".join(f"{r['f_opt']:<13.4f}" for r in res))
print(f"{'Iterations:':<32} " +
      " ".join(f"{r['iterations']:<13}" for r in res))
print(f"{'Converged:':<32} " +
      " ".join(f"{str(r['converged']):<13}" for r in res))
print(f"{'Final ||grad||:':<32} " +
      " ".join(f"{r['grad_norms'][-1]:<13.2e}" for r in res))
if 'n_active' in result_newton:
    print(f"{'Newton variables:':<32} " +
          " ".join(["n/a          ", "n/a          ", "n/a          ",
                    f"{2*result_newton['n_active']:<13}"]))

# ── Convergence Plot ──────────────────────────────────────────────────────────
COLORS = {'GD': '#3498db', 'L-BFGS': '#e67e22', 'SLSQP': '#e74c3c', 'Newton': '#2ecc71'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Solver Convergence Comparison  —  GD | L-BFGS | SLSQP | Newton',
             fontsize=13, fontweight='bold')

newton_offset = result_slsqp['iterations']

for ax, key, ylabel in [
    (ax1, 'history',    'Objective Value f(x,t)'),
    (ax2, 'grad_norms', 'Gradient Norm ||∇f||'),
]:
    ax.semilogy(result_gd[key],    color=COLORS['GD'],
                linewidth=2, alpha=0.85, label=f"GD ({len(result_gd[key])} iters)")
    ax.semilogy(result_lbfgs[key], color=COLORS['L-BFGS'],
                linewidth=2, alpha=0.85, label=f"L-BFGS ({len(result_lbfgs[key])} iters)")
    ax.semilogy(result_slsqp[key], color=COLORS['SLSQP'],
                linewidth=2, alpha=0.85, label=f"SLSQP ({len(result_slsqp[key])} iters)")
    n_iters = np.arange(newton_offset, newton_offset + len(result_newton[key]))
    ax.semilogy(n_iters, result_newton[key], color=COLORS['Newton'],
                linewidth=2, marker='o', markersize=4,
                label=f"Newton ({len(result_newton[key])} iters, reduced)")
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax1.set_title('Objective Value')
ax2.set_title('Gradient Norm')

plt.tight_layout()
out_dir  = os.path.join(BASE, '..', 'outputs')
out_path = os.path.join(out_dir, 'all_solvers_comparison.png')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nConvergence plot saved to outputs/all_solvers_comparison.png")
print(SEP)
print("All tests completed!")
print(SEP)
