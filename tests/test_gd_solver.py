"""
Test script for Gradient Descent with Armijo backtracking solver.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import sys
sys.path.append('src')
from solvers import gradient_descent_armijo, compute_gradient

# Load data (same as Cell 1 of app.ipynb)
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
    """Wrapper for objective function."""
    x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t = z[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

    v_t = smooth_traffic_curve(t)
    tau = np.clip(1.0 - (v_t - 27.0) / 9.0, 0.0, 1.0)

    transport_cost = np.sum(D_penalized * (1.0 + ALPHA * tau**2) * x)
    excess = np.maximum(0.0, np.sum(x, axis=0) - Q_j)
    capacity_pen = MU * np.sum(excess**2)

    return (transport_cost + capacity_pen) / 10000.0


def gradient_function(z):
    """Wrapper for gradient function."""
    return compute_gradient(z, D_penalized, Q_j, N_DISTRICTS, N_STATIONS,
                           N_ROUTES, ALPHA, MU, smooth_traffic_curve)


# Initialize: each district sends to nearest same-side station
x0_matrix = np.zeros((N_DISTRICTS, N_STATIONS))
for i, d_name in enumerate(districts_list):
    costs = D_penalized[i].copy()
    x0_matrix[i, int(np.argmin(costs))] = d_i[i]

z0 = np.concatenate([x0_matrix.flatten(), np.full(N_ROUTES, 14.0)])

print("="*60)
print("TESTING GRADIENT DESCENT WITH ARMIJO BACKTRACKING")
print("="*60)
print(f"Problem size: {N_DISTRICTS} districts × {N_STATIONS} stations = {N_ROUTES} routes")
print(f"Total variables: {2 * N_ROUTES}")
print(f"Parameters: α={ALPHA}, μ={MU}, crossing_penalty={CROSSING_PENALTY}")
print(f"Initial objective: {objective_function(z0):.6f}")
print("="*60)

# Run gradient descent
result = gradient_descent_armijo(
    z0=z0,
    objective_fn=objective_function,
    gradient_fn=gradient_function,
    d_i=d_i,
    N_DISTRICTS=N_DISTRICTS,
    N_STATIONS=N_STATIONS,
    N_ROUTES=N_ROUTES,
    alpha_init=1.0,
    beta=0.5,
    c=1e-4,
    max_iter=200,  # Reduced for testing
    tol=1e-5,
    verbose=True
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"Final objective: {result['f_opt']:.6f}")
print(f"Initial objective: {result['history'][0]:.6f}")
print(f"Improvement: {result['history'][0] - result['f_opt']:.6f}")
print(f"Final ||∇f||: {result['grad_norms'][-1]:.6e}")

# Check constraint satisfaction
z_opt = result['z_opt']
x_opt = z_opt[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
district_sums = np.sum(x_opt, axis=1)
constraint_violations = np.abs(district_sums - d_i)
print(f"\nConstraint satisfaction:")
print(f"  Max |Σ_j x_ij - d_i|: {np.max(constraint_violations):.6e}")
print(f"  Mean |Σ_j x_ij - d_i|: {np.mean(constraint_violations):.6e}")

# Check capacity overflow
station_loads = np.sum(x_opt, axis=0)
overload = np.sum(np.maximum(0.0, station_loads - Q_j))
print(f"\nCapacity overflow: {overload:.1f} tons")

print("\n✅ Test completed successfully!")
