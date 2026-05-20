# Istanbul Municipal Waste Routing Optimization

**YZV 202E — Optimization for Data Science Project**
**Istanbul Technical University**

A traffic-aware waste allocation and departure time optimization system for Istanbul's 39 districts and 9 transfer stations. Minimizes transport costs while respecting capacity constraints and preventing Bosphorus crossings.

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Option 1: Interactive Notebook (Recommended)

```bash
jupyter notebook app.ipynb
```

**Usage:**
1. Run Cell 1: Loads district and station data
2. Run Cell 2: Launches interactive optimization interface
3. Adjust sliders:
   - **α (Traffic Weight)**: 0-5 (default: 0.5) — sensitivity to traffic congestion
   - **μ (Capacity Penalty)**: 50-2000 (default: 500) — penalty for exceeding station capacity
   - **Bosphorus Penalty**: 10-500× (default: 100×) — multiplier for cross-Bosphorus routes
4. Click "Run Interact" to optimize and view results
5. Interactive map opens automatically in your browser

### Option 2: Run Test Scripts

```bash
# Compare all three solvers (GD, SLSQP, Newton)
python test_all_solvers.py

# Complete pipeline with post-processing
python test_optimal_pipeline.py

# Stochastic demand robustness test
python test_stochastic.py

# Unit test: gradient correctness
python test_gradient_unit.py
```

## Project Structure

```
├── app.ipynb                   # Interactive notebook (main interface)
├── src/
│   ├── solvers.py              # All optimization algorithms + post-processing
│   ├── model.py                # SLSQP solver module
│   ├── data_prep.py            # Distance matrix generation
│   └── visualizations.py       # Map generation
├── data/
│   ├── ilceler.xlsx            # District waste data (2004-2025)
│   ├── istasyonlar.xlsx        # Station locations and capacities
│   ├── trafik.xlsx             # Hourly traffic speeds
│   ├── ilce_koordinat.xlsx     # District coordinates + sides
│   └── distance_matrix.npy     # 39×9 OSRM road distances
├── test_*.py                   # Test suite
└── outputs/                    # Generated plots and results
```

## Problem Formulation

**Decision Variables** (702 total):
- `x_ij ≥ 0`: tons/day from district i to station j
- `t_ij ∈ [6, 22]`: departure hour for route (i,j)

**Objective Function** (Proposal Eq. 2):
```
min f(x,t) = Σ_ij D_ij(1 + α·τ(t_ij)²)·x_ij  +  μ·Σ_j[max(0, Σ_i x_ij - Q_j)]²
```

Where:
- `D_ij`: road distance (km) with 100× penalty for Bosphorus crossings
- `τ(t)`: congestion index from traffic data (0 = free-flow, 1 = peak congestion)
- `α`: traffic sensitivity parameter
- `μ`: capacity violation penalty weight

**Constraints:**
- `Σ_j x_ij = d_i` ∀i (all waste transported daily)
- `x_ij ≥ 0`, `t_ij ∈ [6, 22]`
- Active routes ≤ 82 (fleet constraint)
- Bosphorus crossings = 0 (enforced via post-processing)

## Optimization Methods

Three algorithms demonstrating different convergence rates:

1. **Gradient Descent with Armijo Backtracking** — Linear convergence
2. **SLSQP (Sequential Least Squares Programming)** — Superlinear convergence (recommended)
3. **Newton's Method on Reduced Problem** — Quadratic convergence

**Post-Processing Pipeline:**
```
SLSQP → Crossing-Aware Pruning → Capacity Rebalancing
```

- **Crossing-Aware Pruning**: Assigns each district to best same-side station (39 routes, 0 crossings)
- **Capacity Rebalancing**: Redistributes from overloaded stations to alternatives

## Results

**Solver Performance** (100 iterations):
- Gradient Descent: 1013.11
- SLSQP: 438.55 (2.3× better)
- Newton (reduced): 275.72 (3.7× better)

**Stochastic Robustness** (10 scenarios, demand ~10% std dev):
- Objective: 20,819 ± 12,674
- Capacity overflow: 1,028 ± 232 tons
- All scenarios respect fleet constraint (79-82 routes)

**Known Limitation:**
European-side capacity is insufficient for European-side demand when Bosphorus crossings are forbidden (~2,900 tons overflow vs. 605 ton tolerance). This is a fundamental problem constraint documented as Issue R1 in PRD. Baruthane station (340 ton/day capacity) is the primary bottleneck serving high-density districts.

## Data Sources

All data from **Istanbul Metropolitan Municipality (IBB) Open Data Portal**:
- District waste: 2004-2025 (22 years)
- Station locations: GPS coordinates + physical areas
- Traffic speeds: Hourly averages (25th percentile, January 2025)
- Distances: OSRM routing API (road network, not Euclidean)

## Output Files

- `data/interaktif_harita.html` — Interactive Folium map
- `outputs/all_solvers_comparison.png` — Convergence plots
- `outputs/station_loads_complete_pipeline.png` — Capacity utilization chart
- `outputs/stochastic_results.csv` — Scenario test results

## Documentation

- `CLAUDE.md` — Developer guide for Claude Code
- `PROJECT_PRD.md` — Detailed requirements document
- `proposal.pdf` — Original project proposal

## Authors

- Erdem Özseven
- Çağan Çobanoğlu
- Mehmet Arda Öncel

**Course:** YZV 202E — Optimization for Data Science
**Institution:** Istanbul Technical University
**Date:** May 2026

## License

Academic project for ITU YZV 202E course.
