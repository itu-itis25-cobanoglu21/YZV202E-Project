"""
Test stochastic demand robustness with perturbed demand scenarios.

Runs the optimizer over N demand scenarios sampled from N(μ_di, σ_di²)
where σ is estimated from 22 years of IBB historical data (2004-2025).

Reports mean ± std of objective value and capacity overflow across scenarios.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from solvers import slsqp_solver, compute_gradient, fleet_constraint_pruning, capacity_aware_rebalancing

# Random seed for reproducibility
np.random.seed(42)

# Load data
df_ilce_ref = pd.read_excel("data/ilce_koordinat.xlsx")
df_trafik = pd.read_excel("data/trafik.xlsx")

df_ilceler_raw = pd.read_excel("data/ilceler.xlsx")
df_ilceler_raw.columns = df_ilceler_raw.columns.astype(str).str.strip()

# Get all available years for computing statistics
all_years = [str(y) for y in range(2004, 2026) if str(y) in df_ilceler_raw.columns]
print(f"Computing demand statistics from {len(all_years)} years: {all_years[0]}–{all_years[-1]}")

# Compute mean and std dev from historical data
df_ilceler_historical = df_ilceler_raw[all_years].copy()
mean_tonnage = df_ilceler_historical.mean(axis=1)
std_tonnage = df_ilceler_historical.std(axis=1)

# Create district dataframe with statistics
df_ilceler_dinamik = pd.DataFrame({
    'Ilce_Adi': df_ilceler_raw['İlçe'].astype(str).str.strip().str.title(),
    'Mean_Yillik_Tonaj': mean_tonnage,
    'Std_Yillik_Tonaj': std_tonnage
})

df_ilceler = pd.merge(df_ilce_ref, df_ilceler_dinamik, on='Ilce_Adi', how='left')

# Fill missing values
df_ilceler['Mean_Yillik_Tonaj'] = df_ilceler['Mean_Yillik_Tonaj'].fillna(
    df_ilceler['Mean_Yillik_Tonaj'].mean())
df_ilceler['Std_Yillik_Tonaj'] = df_ilceler['Std_Yillik_Tonaj'].fillna(
    df_ilceler['Std_Yillik_Tonaj'].mean())

# Load station data
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

# Convert to daily with normalization
mu_di = (df_ilceler['Mean_Yillik_Tonaj'] / 365.0).to_numpy()
sigma_di = (df_ilceler['Std_Yillik_Tonaj'] / 365.0).to_numpy()

# Normalize mean to Q_TOTAL
mu_di = mu_di * (Q_TOTAL / mu_di.sum())
sigma_di = sigma_di * (Q_TOTAL / mu_di.sum())  # Scale std proportionally

df_ilceler['Gunluk_Tonaj_Mean'] = mu_di
df_ilceler['Gunluk_Tonaj_Std'] = sigma_di

# Setup problem
N_DISTRICTS = len(df_ilceler)
N_STATIONS = len(df_istasyonlar)
N_ROUTES = N_DISTRICTS * N_STATIONS

districts_list = df_ilceler['Ilce_Adi'].tolist()
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


# Stochastic simulation
N_SCENARIOS = 10
MAX_ITER_PER_SCENARIO = 50  # Reduced for speed

print("="*80)
print("STOCHASTIC DEMAND ROBUSTNESS TEST")
print("="*80)
print(f"Number of scenarios: {N_SCENARIOS}")
print(f"Max iterations per scenario: {MAX_ITER_PER_SCENARIO}")
print(f"Mean daily demand: {mu_di.sum():.1f} ton/day")
print(f"Demand std (average across districts): {sigma_di.mean():.2f} ton/day")
print("="*80)

scenario_results = []

for scenario_idx in range(N_SCENARIOS):
    print(f"\nScenario {scenario_idx + 1}/{N_SCENARIOS}")
    print("-" * 60)

    # Sample demand from N(μ_di, σ_di²)
    d_i_scenario = np.random.normal(mu_di, sigma_di)
    d_i_scenario = np.maximum(0.0, d_i_scenario)  # Ensure non-negative

    # Renormalize to preserve total tonnage
    d_i_scenario *= (Q_TOTAL / d_i_scenario.sum())

    print(f"  Sampled demand range: [{d_i_scenario.min():.1f}, {d_i_scenario.max():.1f}] ton/day")

    # Initialize
    x0_matrix = np.zeros((N_DISTRICTS, N_STATIONS))
    for i, d_name in enumerate(districts_list):
        costs = D_penalized[i].copy()
        x0_matrix[i, int(np.argmin(costs))] = d_i_scenario[i]

    z0 = np.concatenate([x0_matrix.flatten(), np.full(N_ROUTES, 14.0)])

    # Run SLSQP solver
    result = slsqp_solver(
        z0=z0,
        objective_fn=objective_function,
        gradient_fn=gradient_function,
        d_i=d_i_scenario,
        N_DISTRICTS=N_DISTRICTS,
        N_STATIONS=N_STATIONS,
        N_ROUTES=N_ROUTES,
        max_iter=MAX_ITER_PER_SCENARIO,
        tol=1e-5,  # Relaxed tolerance for speed
        verbose=False
    )

    z_opt = result['z_opt']
    x_opt = z_opt[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)

    # Apply fleet constraint
    x_fleet = fleet_constraint_pruning(
        x_opt, d_i_scenario, N_DISTRICTS, N_STATIONS,
        epsilon=5.0, max_routes=82
    )

    # Apply capacity rebalancing
    x_final = capacity_aware_rebalancing(
        x_fleet, Q_j, D_penalized,
        districts_yaka, stations_yaka,
        N_DISTRICTS, N_STATIONS,
        capacity_tolerance=1.05
    )

    # Reconstruct z with final x
    z_final = np.concatenate([x_final.flatten(), z_opt[N_ROUTES:]])
    f_final = objective_function(z_final)

    # Check capacity overflow
    station_loads = np.sum(x_final, axis=0)
    overflow = np.sum(np.maximum(0.0, station_loads - Q_j))
    n_active = np.sum(x_final > 0)

    print(f"  Final objective: {f_final:.2f}")
    print(f"  Active routes: {n_active} (limit: 82)")
    print(f"  Capacity overflow: {overflow:.1f} tons")

    scenario_results.append({
        'scenario': scenario_idx + 1,
        'objective': f_final,
        'overflow': overflow,
        'n_active': n_active,
        'converged': result['converged']
    })

# Summary statistics
results_df = pd.DataFrame(scenario_results)

print("\n" + "="*80)
print("STOCHASTIC ROBUSTNESS SUMMARY")
print("="*80)
print(f"{'Metric':<30} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
print("-"*80)
print(f"{'Objective f(x,t):':<30} {results_df['objective'].mean():<15.2f} "
      f"{results_df['objective'].std():<15.2f} {results_df['objective'].min():<15.2f} "
      f"{results_df['objective'].max():<15.2f}")
print(f"{'Capacity overflow (tons):':<30} {results_df['overflow'].mean():<15.1f} "
      f"{results_df['overflow'].std():<15.1f} {results_df['overflow'].min():<15.1f} "
      f"{results_df['overflow'].max():<15.1f}")
print(f"{'Active routes:':<30} {results_df['n_active'].mean():<15.1f} "
      f"{results_df['n_active'].std():<15.2f} {results_df['n_active'].min():<15.0f} "
      f"{results_df['n_active'].max():<15.0f}")

convergence_rate = results_df['converged'].sum() / N_SCENARIOS * 100
print(f"\nConvergence rate: {convergence_rate:.1f}% ({results_df['converged'].sum()}/{N_SCENARIOS} scenarios)")

# Coefficient of variation
cv_objective = results_df['objective'].std() / results_df['objective'].mean()
cv_overflow = results_df['overflow'].std() / (results_df['overflow'].mean() + 1e-6)

print(f"\nCoefficient of Variation:")
print(f"  Objective: {cv_objective:.3f} ({'low variance' if cv_objective < 0.1 else 'moderate variance' if cv_objective < 0.2 else 'high variance'})")
print(f"  Overflow:  {cv_overflow:.3f} ({'low variance' if cv_overflow < 0.5 else 'moderate variance' if cv_overflow < 1.0 else 'high variance'})")

print("\n" + "="*80)
print("✅ Stochastic demand test completed!")
print("="*80)

# Save results
os.makedirs('outputs', exist_ok=True)
results_df.to_csv('outputs/stochastic_results.csv', index=False)
print(f"\nResults saved to outputs/stochastic_results.csv")
