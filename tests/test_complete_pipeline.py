"""
Complete optimization pipeline test with all features:
- SLSQP solver
- Fleet constraint (ε-pruning)
- Capacity-aware rebalancing
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from solvers import (slsqp_solver, compute_gradient, fleet_constraint_pruning,
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
MU = 500.0
CROSSING_PENALTY = 100.0

# Build penalized distance matrix
D_penalized = D.copy()
D_penalized[CROSSING_MASK] *= CROSSING_PENALTY

# Get side labels for rebalancing
districts_yaka = df_ilceler['Yaka'].to_numpy()
stations_yaka = df_istasyonlar['Yaka'].to_numpy()


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


# Initialize: each district sends to nearest same-side station
x0_matrix = np.zeros((N_DISTRICTS, N_STATIONS))
for i, d_name in enumerate(districts_list):
    costs = D_penalized[i].copy()
    x0_matrix[i, int(np.argmin(costs))] = d_i[i]

z0 = np.concatenate([x0_matrix.flatten(), np.full(N_ROUTES, 14.0)])

print("="*80)
print("COMPLETE OPTIMIZATION PIPELINE TEST")
print("="*80)
print(f"Problem: {N_DISTRICTS} districts × {N_STATIONS} stations = {N_ROUTES} routes")
print(f"Total variables: {2 * N_ROUTES}")
print(f"Initial objective: {objective_function(z0):.2f}")
print("="*80)

# Step 1: Run SLSQP solver
print("\n" + "="*80)
print("STEP 1: SLSQP OPTIMIZATION")
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

z_opt = result_slsqp['z_opt']
x_opt = z_opt[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
t_opt = z_opt[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

# Check initial state
station_loads_initial = np.sum(x_opt, axis=0)
overflow_initial = np.sum(np.maximum(0.0, station_loads_initial - Q_j))
n_active_initial = np.sum(x_opt > 0)

print(f"\nAfter SLSQP:")
print(f"  Objective: {result_slsqp['f_opt']:.2f}")
print(f"  Active routes: {n_active_initial}")
print(f"  Capacity overflow: {overflow_initial:.1f} tons")

# Step 2: Apply fleet constraint
print("\n" + "="*80)
print("STEP 2: FLEET CONSTRAINT (ε-PRUNING)")
print("="*80)
print("Pruning routes with x_ij < 5.0 tons/day, enforcing ≤82 active routes")

x_fleet = fleet_constraint_pruning(
    x_opt, d_i, N_DISTRICTS, N_STATIONS,
    epsilon=5.0, max_routes=82
)

z_fleet = np.concatenate([x_fleet.flatten(), t_opt.flatten()])
f_fleet = objective_function(z_fleet)

station_loads_fleet = np.sum(x_fleet, axis=0)
overflow_fleet = np.sum(np.maximum(0.0, station_loads_fleet - Q_j))
n_active_fleet = np.sum(x_fleet > 0)

print(f"\nAfter fleet constraint:")
print(f"  Objective: {f_fleet:.2f}")
print(f"  Active routes: {n_active_fleet} (limit: 82)")
print(f"  Capacity overflow: {overflow_fleet:.1f} tons")

# Step 3: Apply crossing-aware pruning
print("\n" + "="*80)
print("STEP 3: CROSSING-AWARE PRUNING")
print("="*80)
print("Assigning each district to single best same-side station (eliminates crossings)")

x_crossing, t_crossing = crossing_aware_pruning(
    x_fleet, t_opt,
    d_i, districts_yaka, stations_yaka,
    N_DISTRICTS, N_STATIONS
)

z_crossing = np.concatenate([x_crossing.flatten(), t_crossing.flatten()])
f_crossing = objective_function(z_crossing)

station_loads_crossing = np.sum(x_crossing, axis=0)
overflow_crossing = np.sum(np.maximum(0.0, station_loads_crossing - Q_j))
n_active_crossing = np.sum(x_crossing > 0)
n_crossings_after = np.sum((x_crossing > 0) & CROSSING_MASK)

print(f"\nAfter crossing-aware pruning:")
print(f"  Objective: {f_crossing:.2f}")
print(f"  Active routes: {n_active_crossing} (1 per district)")
print(f"  Capacity overflow: {overflow_crossing:.1f} tons")
print(f"  Bosphorus crossings: {n_crossings_after} (target: 0)")

# Step 4: Apply capacity-aware rebalancing
print("\n" + "="*80)
print("STEP 4: CAPACITY-AWARE REBALANCING")
print("="*80)
print("Reassigning largest contributors from overloaded stations (>105% capacity)")

x_final = capacity_aware_rebalancing(
    x_crossing, Q_j, D_penalized,
    districts_yaka, stations_yaka,
    N_DISTRICTS, N_STATIONS,
    capacity_tolerance=1.05
)

z_final = np.concatenate([x_final.flatten(), t_crossing.flatten()])
f_final = objective_function(z_final)

station_loads_final = np.sum(x_final, axis=0)
overflow_final = np.sum(np.maximum(0.0, station_loads_final - Q_j))
n_active_final = np.sum(x_final > 0)
n_crossings_final = np.sum((x_final > 0) & CROSSING_MASK)

print(f"\nAfter capacity rebalancing:")
print(f"  Objective: {f_final:.2f}")
print(f"  Active routes: {n_active_final}")
print(f"  Capacity overflow: {overflow_final:.1f} tons")
print(f"  Bosphorus crossings: {n_crossings_final}")

# Summary table
print("\n" + "="*80)
print("PIPELINE SUMMARY")
print("="*80)
print(f"{'Stage':<35} {'Objective':<12} {'Routes':<10} {'Overflow':<12} {'Crossings':<12}")
print("-"*80)
print(f"{'1. SLSQP Optimization':<35} {result_slsqp['f_opt']:<12.2f} {n_active_initial:<10} {overflow_initial:<12.1f} {'-':<12}")
print(f"{'2. Fleet Constraint (≤82)':<35} {f_fleet:<12.2f} {n_active_fleet:<10} {overflow_fleet:<12.1f} {'-':<12}")
print(f"{'3. Crossing-Aware Pruning':<35} {f_crossing:<12.2f} {n_active_crossing:<10} {overflow_crossing:<12.1f} {n_crossings_after:<12}")
print(f"{'4. Capacity Rebalancing':<35} {f_final:<12.2f} {n_active_final:<10} {overflow_final:<12.1f} {n_crossings_final:<12}")

# Per-station breakdown
print("\n" + "="*80)
print("STATION CAPACITY UTILIZATION")
print("="*80)
print(f"{'Station':<40} {'Capacity':<12} {'Load':<12} {'Utilization':<12}")
print("-"*80)
for j in range(N_STATIONS):
    station_name = station_names[j][:37]  # Truncate long names
    capacity = Q_j[j]
    load = station_loads_final[j]
    util = load / capacity * 100

    status = "✅" if load <= capacity * 1.05 else "⚠️"
    print(f"{status} {station_name:<38} {capacity:<12.1f} {load:<12.1f} {util:<12.1f}%")

# Check constraint violations
district_sums = np.sum(x_final, axis=1)
constraint_violations = np.abs(district_sums - d_i)
max_violation = np.max(constraint_violations)

print(f"\nConstraint satisfaction:")
print(f"  Max |Σ_j x_ij - d_i|: {max_violation:.6e}")
print(f"  All districts satisfied: {'Yes ✅' if max_violation < 1e-3 else 'No ❌'}")

# Visualize station loads
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(N_STATIONS)
capacities = Q_j
loads = station_loads_final

colors = ['green' if l <= c * 1.05 else 'orange' for l, c in zip(loads, capacities)]

ax.bar(x_pos, capacities, alpha=0.4, color='gray', label='Capacity', width=0.6)
ax.bar(x_pos, loads, alpha=0.8, color=colors, label='Load', width=0.6)

ax.axhline(y=Q_TOTAL / N_STATIONS, color='red', linestyle='--', alpha=0.5, label='Average capacity')

ax.set_xlabel('Station', fontsize=11)
ax.set_ylabel('Tonnage (ton/day)', fontsize=11)
ax.set_title('Station Capacity vs. Load (After Complete Pipeline)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([name[:20] for name in station_names], rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/station_loads_complete_pipeline.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Station load chart saved to outputs/station_loads_complete_pipeline.png")

print("\n" + "="*80)
print("Complete pipeline test finished!")
print("="*80)
