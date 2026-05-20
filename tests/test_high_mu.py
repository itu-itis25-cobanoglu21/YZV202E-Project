"""
Test effect of high capacity penalty (μ) on final solution quality.

Compares μ=500 (baseline) vs μ=2000 vs μ=5000.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from solvers import (slsqp_solver, compute_gradient,
                     capacity_aware_rebalancing, crossing_aware_pruning)

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
CROSSING_PENALTY = 100.0

# Build penalized distance matrix
D_penalized = D.copy()
D_penalized[CROSSING_MASK] *= CROSSING_PENALTY

# Get side labels
districts_yaka = df_ilceler['Yaka'].to_numpy()
stations_yaka = df_istasyonlar['Yaka'].to_numpy()


def test_mu_value(MU_value):
    """Run complete pipeline with given μ value."""

    def objective_function(z):
        x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
        t = z[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

        v_t = smooth_traffic_curve(t)
        tau = np.clip(1.0 - (v_t - 27.0) / 9.0, 0.0, 1.0)

        transport_cost = np.sum(D_penalized * (1.0 + ALPHA * tau**2) * x)
        excess = np.maximum(0.0, np.sum(x, axis=0) - Q_j)
        capacity_pen = MU_value * np.sum(excess**2)

        return (transport_cost + capacity_pen) / 10000.0

    def gradient_function(z):
        return compute_gradient(z, D_penalized, Q_j, N_DISTRICTS, N_STATIONS,
                               N_ROUTES, ALPHA, MU_value, smooth_traffic_curve)

    # Initialize
    x0_matrix = np.zeros((N_DISTRICTS, N_STATIONS))
    for i, d_name in enumerate(districts_list):
        costs = D_penalized[i].copy()
        x0_matrix[i, int(np.argmin(costs))] = d_i[i]

    z0 = np.concatenate([x0_matrix.flatten(), np.full(N_ROUTES, 14.0)])

    # SLSQP
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
        verbose=False
    )

    z_opt = result_slsqp['z_opt']
    x_opt = z_opt[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t_opt = z_opt[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

    overflow_slsqp = np.sum(np.maximum(0.0, np.sum(x_opt, axis=0) - Q_j))

    # Crossing-aware pruning
    x_crossing, t_crossing = crossing_aware_pruning(
        x_opt, t_opt,
        d_i, districts_yaka, stations_yaka,
        N_DISTRICTS, N_STATIONS
    )

    overflow_crossing = np.sum(np.maximum(0.0, np.sum(x_crossing, axis=0) - Q_j))

    # Capacity rebalancing
    x_final = capacity_aware_rebalancing(
        x_crossing, Q_j, D_penalized,
        districts_yaka, stations_yaka,
        N_DISTRICTS, N_STATIONS,
        capacity_tolerance=1.05
    )

    z_final = np.concatenate([x_final.flatten(), t_crossing.flatten()])
    f_final = objective_function(z_final)
    overflow_final = np.sum(np.maximum(0.0, np.sum(x_final, axis=0) - Q_j))
    n_crossings = np.sum((x_final > 0) & CROSSING_MASK)

    return {
        'mu': MU_value,
        'f_slsqp': result_slsqp['f_opt'],
        'overflow_slsqp': overflow_slsqp,
        'overflow_crossing': overflow_crossing,
        'overflow_final': overflow_final,
        'f_final': f_final,
        'n_crossings': n_crossings,
        'converged': result_slsqp['converged']
    }


print("="*80)
print("HIGH μ (CAPACITY PENALTY) TEST")
print("="*80)
print("Testing μ values: 500, 2000, 5000")
print("Pipeline: SLSQP → Crossing-Aware → Rebalancing")
print("="*80)

results = []
for mu_val in [500, 2000, 5000]:
    print(f"\nTesting μ = {mu_val}...")
    result = test_mu_value(mu_val)
    results.append(result)
    print(f"  SLSQP overflow: {result['overflow_slsqp']:.1f} tons")
    print(f"  Final overflow: {result['overflow_final']:.1f} tons")
    print(f"  Crossings: {result['n_crossings']}")

print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)
print(f"{'μ':<10} {'SLSQP Obj':<12} {'SLSQP Flow':<12} {'Final Obj':<12} {'Final Flow':<12} {'Crossings':<10}")
print("-"*80)

for r in results:
    print(f"{r['mu']:<10} {r['f_slsqp']:<12.2f} {r['overflow_slsqp']:<12.1f} "
          f"{r['f_final']:<12.2f} {r['overflow_final']:<12.1f} {r['n_crossings']:<10}")

print("\n" + "="*80)
print(f"5% Capacity Tolerance: {Q_TOTAL * 0.05:.1f} tons")
print("="*80)

best_result = min(results, key=lambda x: x['overflow_final'])
print(f"\nBest result: μ = {best_result['mu']}")
print(f"  Final overflow: {best_result['overflow_final']:.1f} tons")
print(f"  Within 5% tolerance: {'Yes ✅' if best_result['overflow_final'] <= Q_TOTAL * 0.05 else 'No ❌'}")

print("\n" + "="*80)
print("Test completed!")
print("="*80)
