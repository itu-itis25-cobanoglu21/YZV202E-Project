"""
Optimization solvers for Istanbul waste routing problem.

Implements three gradient-based methods demonstrating different convergence rates:
- Gradient Descent with Armijo backtracking line search (linear convergence)
- SLSQP - Sequential Least Squares Programming (quasi-Newton, superlinear convergence)
- Newton's method on reduced problem (quadratic convergence)

The progression from GD → SLSQP → Newton demonstrates increasingly sophisticated
use of curvature information (no Hessian → BFGS approximation → full Hessian).
"""

import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional
from scipy.optimize import minimize
from scipy.linalg import solve


def compute_gradient(z: np.ndarray,
                     D_penalized: np.ndarray,
                     Q_j: np.ndarray,
                     N_DISTRICTS: int,
                     N_STATIONS: int,
                     N_ROUTES: int,
                     ALPHA: float,
                     MU: float,
                     smooth_traffic_curve) -> np.ndarray:
    """
    Compute analytical gradient of objective function.

    Objective: f(x,t) = Σ_ij D_ij(1 + α·τ(t_ij)²)·x_ij + μ·Σ_j[max(0, Σ_i x_ij - Q_j)]²

    Args:
        z: Decision vector [x_flat, t_flat] of length 2*N_ROUTES
        D_penalized: Distance matrix with Bosphorus penalty applied
        Q_j: Station capacities
        N_DISTRICTS: Number of districts (39)
        N_STATIONS: Number of stations (9)
        N_ROUTES: N_DISTRICTS × N_STATIONS (351)
        ALPHA: Traffic sensitivity parameter
        MU: Capacity violation penalty weight
        smooth_traffic_curve: CubicSpline object for v(t)

    Returns:
        grad: Gradient vector same shape as z
    """
    x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t = z[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

    # Traffic speed and congestion index
    v_t = smooth_traffic_curve(t)
    tau = np.clip(1.0 - (v_t - 27.0) / 9.0, 0.0, 1.0)

    # Station loads and excess
    station_loads = np.sum(x, axis=0)  # shape: (N_STATIONS,)
    excess = np.maximum(0.0, station_loads - Q_j)

    # --- Gradient w.r.t. x ---
    # ∂f/∂x_ij = D_ij(1 + α·τ²) + 2μ·max(0, load_j - Q_j)
    grad_x = D_penalized * (1.0 + ALPHA * tau**2)

    # Add capacity penalty gradient (broadcast across districts)
    capacity_grad = 2.0 * MU * excess  # shape: (N_STATIONS,)
    grad_x += capacity_grad[np.newaxis, :]  # broadcast to (N_DISTRICTS, N_STATIONS)

    # --- Gradient w.r.t. t ---
    # ∂f/∂t_ij = D_ij · x_ij · α · 2·τ · ∂τ/∂t
    # τ(t) = clip(1 - (v(t) - v_min)/(v_max - v_min), 0, 1)
    # ∂τ/∂t = -(1/9.0) · dv/dt when τ ∈ (0,1), else 0 (due to clipping)

    dv_dt = smooth_traffic_curve.derivative()(t)  # CubicSpline derivative
    dtau_dt = -(1.0 / 9.0) * dv_dt

    # Zero out gradient where tau is clipped
    tau_unclipped = 1.0 - (v_t - 27.0) / 9.0
    is_clipped = (tau_unclipped <= 0.0) | (tau_unclipped >= 1.0)
    dtau_dt[is_clipped] = 0.0

    grad_t = D_penalized * x * ALPHA * 2.0 * tau * dtau_dt

    # Combine and flatten
    grad = np.concatenate([grad_x.flatten(), grad_t.flatten()])

    # Scale by same factor as objective (1/10000)
    return grad / 10000.0


def project_to_simplex(x: np.ndarray, d_i: np.ndarray, N_DISTRICTS: int, N_STATIONS: int) -> np.ndarray:
    """
    Project x onto feasible simplex to enforce equality constraint Σ_j x_ij = d_i.

    For each district i, redistribute x[i,:] so that it sums to d_i while keeping x_ij ≥ 0.
    Uses efficient simplex projection algorithm.

    Args:
        x: Allocation matrix (N_DISTRICTS, N_STATIONS)
        d_i: Daily waste per district (N_DISTRICTS,)
        N_DISTRICTS: Number of districts
        N_STATIONS: Number of stations

    Returns:
        x_proj: Projected matrix satisfying Σ_j x_ij = d_i
    """
    x_proj = x.reshape(N_DISTRICTS, N_STATIONS).copy()

    for i in range(N_DISTRICTS):
        # Simplex projection for district i: project onto {x : Σx = d_i, x ≥ 0}
        y = x_proj[i, :]
        target_sum = d_i[i]

        if np.sum(y) == target_sum and np.all(y >= 0):
            continue  # Already feasible

        # Sort in descending order
        y_sorted = np.sort(y)[::-1]
        cumsum = np.cumsum(y_sorted)

        # Find rho: largest index where y_sorted[rho] - (cumsum[rho] - target_sum)/(rho+1) > 0
        rho = 0
        for j in range(N_STATIONS):
            theta = (cumsum[j] - target_sum) / (j + 1)
            if y_sorted[j] - theta > 0:
                rho = j
            else:
                break

        theta = (cumsum[rho] - target_sum) / (rho + 1)
        x_proj[i, :] = np.maximum(y - theta, 0.0)

    return x_proj.flatten()


def armijo_backtracking(z: np.ndarray,
                       grad: np.ndarray,
                       objective_fn: Callable,
                       alpha_init: float = 1.0,
                       beta: float = 0.5,
                       c: float = 1e-4,
                       max_backtracks: int = 30) -> float:
    """
    Armijo backtracking line search.

    Find step size α such that:
    f(z - α·∇f) ≤ f(z) - c·α·||∇f||²

    Args:
        z: Current point
        grad: Gradient at z
        objective_fn: Function that takes z and returns f(z)
        alpha_init: Initial step size
        beta: Reduction factor (< 1)
        c: Armijo constant (typically 1e-4)
        max_backtracks: Maximum backtracking iterations

    Returns:
        alpha: Step size satisfying Armijo condition
    """
    f_current = objective_fn(z)
    grad_norm_sq = np.dot(grad, grad)
    alpha = alpha_init

    for _ in range(max_backtracks):
        z_new = z - alpha * grad
        f_new = objective_fn(z_new)

        # Armijo condition
        if f_new <= f_current - c * alpha * grad_norm_sq:
            return alpha

        alpha *= beta

    # If backtracking fails, return smallest alpha tried
    return alpha


def gradient_descent_armijo(z0: np.ndarray,
                            objective_fn: Callable,
                            gradient_fn: Callable,
                            d_i: np.ndarray,
                            N_DISTRICTS: int,
                            N_STATIONS: int,
                            N_ROUTES: int,
                            alpha_init: float = 1.0,
                            beta: float = 0.5,
                            c: float = 1e-4,
                            max_iter: int = 500,
                            tol: float = 1e-6,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Gradient Descent with Armijo backtracking line search.

    Enforces equality constraint Σ_j x_ij = d_i via simplex projection after each step.

    Args:
        z0: Initial point [x_flat, t_flat]
        objective_fn: Function taking z and returning scalar objective value
        gradient_fn: Function taking z and returning gradient vector
        d_i: Daily waste per district (for projection)
        N_DISTRICTS: Number of districts
        N_STATIONS: Number of stations
        N_ROUTES: N_DISTRICTS × N_STATIONS
        alpha_init: Initial step size for Armijo
        beta: Backtracking reduction factor
        c: Armijo constant
        max_iter: Maximum iterations
        tol: Convergence tolerance on ||∇f||
        verbose: Print progress

    Returns:
        result: Dictionary with keys:
            - z_opt: Optimal solution
            - f_opt: Optimal objective value
            - history: List of objective values per iteration
            - grad_norms: List of gradient norms per iteration
            - iterations: Number of iterations
            - converged: Whether convergence was achieved
    """
    z = z0.copy()
    history = []
    grad_norms = []

    if verbose:
        print("Starting Gradient Descent with Armijo backtracking...")

    for iteration in range(max_iter):
        # Compute gradient
        grad = gradient_fn(z)
        grad_norm = np.linalg.norm(grad)
        grad_norms.append(grad_norm)

        # Evaluate objective
        f_val = objective_fn(z)
        history.append(f_val)

        if verbose and iteration % 50 == 0:
            print(f"  Iter {iteration:4d}: f = {f_val:.6f}, ||∇f|| = {grad_norm:.6e}")

        # Check convergence
        if grad_norm < tol:
            if verbose:
                print(f"✅ Converged at iteration {iteration}: ||∇f|| = {grad_norm:.6e} < {tol}")
            return {
                'z_opt': z,
                'f_opt': f_val,
                'history': history,
                'grad_norms': grad_norms,
                'iterations': iteration,
                'converged': True
            }

        # Armijo line search
        alpha = armijo_backtracking(z, grad, objective_fn, alpha_init, beta, c)

        # Gradient step
        z_new = z - alpha * grad

        # Project x onto simplex (enforce Σ_j x_ij = d_i)
        x_new = z_new[:N_ROUTES]
        x_projected = project_to_simplex(x_new, d_i, N_DISTRICTS, N_STATIONS)
        z_new[:N_ROUTES] = x_projected

        # Clip x ≥ 0
        z_new[:N_ROUTES] = np.maximum(z_new[:N_ROUTES], 0.0)

        # Clip t ∈ [6, 22]
        z_new[N_ROUTES:] = np.clip(z_new[N_ROUTES:], 6.0, 22.0)

        z = z_new

    # Max iterations reached
    f_final = objective_fn(z)
    if verbose:
        print(f"⚠️  Max iterations ({max_iter}) reached. Final ||∇f|| = {grad_norms[-1]:.6e}")

    return {
        'z_opt': z,
        'f_opt': f_final,
        'history': history,
        'grad_norms': grad_norms,
        'iterations': max_iter,
        'converged': False
    }


def slsqp_solver(z0: np.ndarray,
                 objective_fn: Callable,
                 gradient_fn: Callable,
                 d_i: np.ndarray,
                 N_DISTRICTS: int,
                 N_STATIONS: int,
                 N_ROUTES: int,
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    SLSQP (Sequential Least Squares Programming) solver.

    SLSQP is a quasi-Newton method that uses BFGS updates to approximate
    the Hessian, achieving superlinear convergence. It natively supports
    equality and inequality constraints.

    This method demonstrates faster convergence than gradient descent
    by incorporating curvature information via the BFGS Hessian approximation.

    Args:
        z0: Initial point [x_flat, t_flat]
        objective_fn: Function taking z and returning scalar objective value
        gradient_fn: Function taking z and returning gradient vector
        d_i: Daily waste per district (for equality constraint)
        N_DISTRICTS: Number of districts
        N_STATIONS: Number of stations
        N_ROUTES: N_DISTRICTS × N_STATIONS
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        result: Dictionary with converged solution
    """
    history = []
    grad_norms = []
    iteration_count = [0]

    def equality_constraint(z):
        """Σ_j x_ij = d_i for all districts."""
        x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
        return np.sum(x, axis=1) - d_i

    def callback(z):
        """Record convergence history."""
        f_val = objective_fn(z)
        grad = gradient_fn(z)
        grad_norm = np.linalg.norm(grad)

        history.append(f_val)
        grad_norms.append(grad_norm)

        if verbose and iteration_count[0] % 20 == 0:
            print(f"  Iter {iteration_count[0]:4d}: f = {f_val:.6f}, ||∇f|| = {grad_norm:.6e}")

        iteration_count[0] += 1

    if verbose:
        print("Starting SLSQP (quasi-Newton with BFGS updates)...")

    # Bounds
    bounds_x = [(0, None) for _ in range(N_ROUTES)]
    bounds_t = [(6.0, 22.0) for _ in range(N_ROUTES)]
    bounds = bounds_x + bounds_t

    # Equality constraint
    constraints = {'type': 'eq', 'fun': equality_constraint}

    # Run SLSQP
    result = minimize(
        objective_fn,
        z0,
        method='SLSQP',
        jac=gradient_fn,
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={
            'maxiter': max_iter,
            'ftol': tol * 1e-3,
            'disp': False
        }
    )

    z_opt = result.x
    f_opt = objective_fn(z_opt)
    final_grad = gradient_fn(z_opt)
    final_grad_norm = np.linalg.norm(final_grad)

    # Add final point to history
    if len(history) == 0 or abs(history[-1] - f_opt) > 1e-6:
        history.append(f_opt)
        grad_norms.append(final_grad_norm)

    if verbose:
        if result.success:
            print(f"✅ Converged: {result.message}")
        else:
            print(f"⚠️  {result.message}")
        print(f"   Final ||∇f|| = {final_grad_norm:.6e}")

    return {
        'z_opt': z_opt,
        'f_opt': f_opt,
        'history': history,
        'grad_norms': grad_norms,
        'iterations': iteration_count[0],
        'converged': result.success
    }


def newton_reduced_problem(z_init: np.ndarray,
                           objective_fn: Callable,
                           gradient_fn: Callable,
                           d_i: np.ndarray,
                           N_DISTRICTS: int,
                           N_STATIONS: int,
                           N_ROUTES: int,
                           epsilon: float = 5.0,
                           max_iter: int = 50,
                           tol: float = 1e-6,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Newton's method on reduced problem (active routes only).

    Takes an initial solution, identifies active routes (x_ij > epsilon),
    and applies Newton's method with full Hessian on this reduced variable set.

    This demonstrates quadratic convergence but is only practical on small problems.
    The full 702-variable problem would require a 702×702 Hessian, which is too
    expensive. By reducing to ~40-80 active variables, Newton's method becomes feasible.

    Args:
        z_init: Initial solution from another solver
        objective_fn: Original objective function
        gradient_fn: Original gradient function
        d_i: Daily waste per district
        N_DISTRICTS: Number of districts
        N_STATIONS: Number of stations
        N_ROUTES: N_DISTRICTS × N_STATIONS
        epsilon: Threshold for active routes (tons/day)
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        result: Dictionary with solution and convergence history
    """
    history = []
    grad_norms = []

    # Step 1: Identify active routes from initial solution
    x_init = z_init[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t_init = z_init[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)

    # For each district, keep top routes that sum to d_i
    active_routes = []
    reduced_x = []
    reduced_t = []
    district_map = []  # Maps reduced index to (i, j)

    for i in range(N_DISTRICTS):
        # Sort routes by tonnage
        route_tonnages = x_init[i, :]
        sorted_indices = np.argsort(route_tonnages)[::-1]

        # Keep routes until we've allocated all waste
        cumsum = 0.0
        for j in sorted_indices:
            if cumsum >= d_i[i] - 0.1:  # Already allocated
                break
            if route_tonnages[j] > epsilon or cumsum < d_i[i]:
                active_routes.append((i, j))
                reduced_x.append(x_init[i, j])
                reduced_t.append(t_init[i, j])
                district_map.append(i)
                cumsum += route_tonnages[j]

    n_active = len(active_routes)
    reduced_x = np.array(reduced_x)
    reduced_t = np.array(reduced_t)
    z_reduced = np.concatenate([reduced_x, reduced_t])

    if verbose:
        print(f"Newton's method on reduced problem...")
        print(f"  Reduced from {2*N_ROUTES} to {2*n_active} variables ({n_active} active routes)")
        print(f"  Active routes per district: {n_active / N_DISTRICTS:.1f}")

    # Step 2: Create reduced objective and gradient functions
    def reduced_objective(z_red):
        """Evaluate objective on reduced problem."""
        # Reconstruct full z with reduced variables
        z_full = z_init.copy()
        for idx, (i, j) in enumerate(active_routes):
            z_full[i * N_STATIONS + j] = z_red[idx]  # x_ij
            z_full[N_ROUTES + i * N_STATIONS + j] = z_red[n_active + idx]  # t_ij
        return objective_fn(z_full)

    def reduced_gradient(z_red):
        """Evaluate gradient on reduced problem."""
        # Reconstruct full z
        z_full = z_init.copy()
        for idx, (i, j) in enumerate(active_routes):
            z_full[i * N_STATIONS + j] = z_red[idx]
            z_full[N_ROUTES + i * N_STATIONS + j] = z_red[n_active + idx]

        # Compute full gradient
        grad_full = gradient_fn(z_full)

        # Extract gradient for active variables
        grad_red = np.zeros(2 * n_active)
        for idx, (i, j) in enumerate(active_routes):
            grad_red[idx] = grad_full[i * N_STATIONS + j]  # ∂f/∂x_ij
            grad_red[n_active + idx] = grad_full[N_ROUTES + i * N_STATIONS + j]  # ∂f/∂t_ij

        return grad_red

    def compute_hessian_fd(z_red, eps=1e-5):
        """Compute Hessian via finite differences."""
        n = len(z_red)
        H = np.zeros((n, n))
        grad_center = reduced_gradient(z_red)

        for i in range(n):
            z_perturb = z_red.copy()
            z_perturb[i] += eps
            grad_perturb = reduced_gradient(z_perturb)
            H[:, i] = (grad_perturb - grad_center) / eps

        # Symmetrize
        H = 0.5 * (H + H.T)
        return H

    # Step 3: Newton's method with backtracking line search
    z = z_reduced.copy()

    for iteration in range(max_iter):
        # Evaluate function and gradient
        f_val = reduced_objective(z)
        grad = reduced_gradient(z)
        grad_norm = np.linalg.norm(grad)

        history.append(f_val)
        grad_norms.append(grad_norm)

        if verbose and iteration % 10 == 0:
            print(f"  Iter {iteration:3d}: f = {f_val:.6f}, ||∇f|| = {grad_norm:.6e}")

        # Check convergence
        if grad_norm < tol:
            if verbose:
                print(f"✅ Converged at iteration {iteration}: ||∇f|| = {grad_norm:.6e}")
            break

        # Compute Hessian
        H = compute_hessian_fd(z)

        # Add regularization if needed (make sure H is positive definite)
        lambda_reg = 1e-6
        while True:
            try:
                # Solve H * d = -grad for Newton direction
                d = solve(H + lambda_reg * np.eye(len(z)), -grad, assume_a='pos')
                break
            except np.linalg.LinAlgError:
                lambda_reg *= 10
                if lambda_reg > 1e-2:
                    # Fall back to gradient descent direction
                    d = -grad
                    break

        # Backtracking line search
        alpha = 1.0
        c = 1e-4
        for _ in range(20):
            z_new = z + alpha * d

            # Enforce bounds
            z_new[:n_active] = np.maximum(z_new[:n_active], 0.0)  # x >= 0
            z_new[n_active:] = np.clip(z_new[n_active:], 6.0, 22.0)  # t in [6, 22]

            f_new = reduced_objective(z_new)

            # Armijo condition
            if f_new <= f_val + c * alpha * np.dot(grad, z_new - z):
                z = z_new
                break

            alpha *= 0.5
        else:
            # Line search failed, take small step
            z = z + 0.01 * d

    # Reconstruct full solution
    z_opt = z_init.copy()
    for idx, (i, j) in enumerate(active_routes):
        z_opt[i * N_STATIONS + j] = z[idx]
        z_opt[N_ROUTES + i * N_STATIONS + j] = z[n_active + idx]

    f_opt = objective_fn(z_opt)

    if verbose:
        print(f"   Final objective: {f_opt:.6f}")
        print(f"   Reduced problem size: {2*n_active} variables")

    return {
        'z_opt': z_opt,
        'f_opt': f_opt,
        'history': history,
        'grad_norms': grad_norms,
        'iterations': len(history),
        'converged': grad_norms[-1] < tol,
        'n_active': n_active
    }


def fleet_constraint_pruning(x: np.ndarray,
                             d_i: np.ndarray,
                             N_DISTRICTS: int,
                             N_STATIONS: int,
                             epsilon: float = 5.0,
                             max_routes: int = 82) -> np.ndarray:
    """
    Enforce fleet constraint: at most K=82 active routes.

    Routes with x_ij < epsilon are pruned (zeroed). Pruned tonnage is redistributed
    to the largest remaining active route for that district to preserve Σ_j x_ij = d_i.

    This is applied BEFORE crossing-aware pruning in the pipeline.

    Args:
        x: Allocation matrix (N_DISTRICTS, N_STATIONS)
        d_i: Daily waste per district (N_DISTRICTS,)
        N_DISTRICTS: Number of districts
        N_STATIONS: Number of stations
        epsilon: Threshold for active routes (tons/day)
        max_routes: Maximum number of active routes (K=82)

    Returns:
        x_pruned: Matrix with small routes zeroed and tonnage redistributed
    """
    x_pruned = x.reshape(N_DISTRICTS, N_STATIONS).copy()

    # Step 1: For each district, keep only routes above epsilon threshold
    # but ensure at least one route per district remains
    for i in range(N_DISTRICTS):
        district_routes = x_pruned[i, :].copy()

        # Find routes above threshold
        above_threshold = district_routes >= epsilon

        if np.sum(above_threshold) == 0:
            # No routes above threshold - keep only the largest
            largest_j = int(np.argmax(district_routes))
            x_pruned[i, :] = 0.0
            x_pruned[i, largest_j] = d_i[i]
        else:
            # Prune routes below threshold, redistribute to largest active route
            below_threshold = district_routes < epsilon
            pruned_tonnage = np.sum(district_routes[below_threshold])

            x_pruned[i, below_threshold] = 0.0

            if pruned_tonnage > 0:
                # Redistribute to largest active route
                largest_j = int(np.argmax(x_pruned[i, :]))
                x_pruned[i, largest_j] += pruned_tonnage

    # Step 2: If still exceeding max_routes, use greedy pruning
    # while ensuring each district keeps at least one route
    n_active = np.sum(x_pruned > 0)

    if n_active > max_routes:
        # For each district, find the best route (highest tonnage or lowest cost)
        # Guarantee these routes are kept
        guaranteed_routes = set()
        for i in range(N_DISTRICTS):
            best_j = int(np.argmax(x_pruned[i, :]))
            guaranteed_routes.add((i, best_j))

        # Collect all other routes and sort by tonnage
        other_routes = []
        for i in range(N_DISTRICTS):
            for j in range(N_STATIONS):
                if x_pruned[i, j] > 0 and (i, j) not in guaranteed_routes:
                    other_routes.append((x_pruned[i, j], i, j))

        other_routes.sort(reverse=True)  # Sort descending

        # Keep top (max_routes - len(guaranteed_routes)) additional routes
        n_additional = max(0, max_routes - len(guaranteed_routes))
        routes_to_keep = guaranteed_routes.copy()
        for tonnage, i, j in other_routes[:n_additional]:
            routes_to_keep.add((i, j))

        # Prune routes not in keep set and redistribute
        for i in range(N_DISTRICTS):
            district_tonnage = 0.0
            routes_to_prune = []

            for j in range(N_STATIONS):
                if x_pruned[i, j] > 0:
                    if (i, j) in routes_to_keep:
                        district_tonnage += x_pruned[i, j]
                    else:
                        routes_to_prune.append(j)

            # Prune and redistribute
            pruned_total = 0.0
            for j in routes_to_prune:
                pruned_total += x_pruned[i, j]
                x_pruned[i, j] = 0.0

            if pruned_total > 0:
                # Redistribute proportionally to remaining routes
                remaining_routes = [j for j in range(N_STATIONS) if x_pruned[i, j] > 0]
                if remaining_routes:
                    # Just add to the largest route for simplicity
                    largest_j = int(np.argmax(x_pruned[i, :]))
                    x_pruned[i, largest_j] += pruned_total

    # Verify constraint satisfaction
    district_sums = np.sum(x_pruned, axis=1)
    max_violation = np.max(np.abs(district_sums - d_i))
    n_active_final = np.sum(x_pruned > 0)

    if max_violation > 1e-3:
        print(f"⚠️  Warning: Fleet pruning constraint violation: max {max_violation:.2e}")
        print(f"   Fixing constraint violations...")
        # Fix any remaining violations
        for i in range(N_DISTRICTS):
            current_sum = np.sum(x_pruned[i, :])
            if abs(current_sum - d_i[i]) > 1e-6:
                # Find the largest route and adjust
                largest_j = int(np.argmax(x_pruned[i, :]))
                x_pruned[i, largest_j] += (d_i[i] - current_sum)
                x_pruned[i, largest_j] = max(0.0, x_pruned[i, largest_j])

    return x_pruned


def capacity_aware_rebalancing(x: np.ndarray,
                               Q_j: np.ndarray,
                               D_penalized: np.ndarray,
                               districts_yaka: np.ndarray,
                               stations_yaka: np.ndarray,
                               N_DISTRICTS: int,
                               N_STATIONS: int,
                               capacity_tolerance: float = 1.05) -> np.ndarray:
    """
    Rebalance allocations to respect station capacity constraints.

    After pruning to 1 route per district, if any station is overloaded beyond
    capacity_tolerance, reassign the largest contributing district to its
    second-best same-side station.

    Args:
        x: Allocation matrix after pruning (N_DISTRICTS, N_STATIONS)
        Q_j: Station capacities (N_STATIONS,)
        D_penalized: Distance matrix with Bosphorus penalty
        districts_yaka: Array of district sides ('Asya' or 'Avrupa')
        stations_yaka: Array of station sides ('Asya' or 'Avrupa')
        N_DISTRICTS: Number of districts
        N_STATIONS: Number of stations
        capacity_tolerance: Station capacity tolerance (default 1.05 = 105%)

    Returns:
        x_rebalanced: Matrix with stations within capacity tolerance
    """
    x_rebal = x.reshape(N_DISTRICTS, N_STATIONS).copy()

    max_iterations = 100
    for iteration in range(max_iterations):
        station_loads = np.sum(x_rebal, axis=0)
        overloaded = station_loads > Q_j * capacity_tolerance

        if not np.any(overloaded):
            break  # All stations within capacity

        # Find most overloaded station
        overflow = station_loads - Q_j * capacity_tolerance
        overflow[~overloaded] = -np.inf
        j_overloaded = int(np.argmax(overflow))

        # Find largest district assigned to this station
        contributors = x_rebal[:, j_overloaded]
        if np.max(contributors) == 0:
            break  # No contributors (shouldn't happen)

        i_largest = int(np.argmax(contributors))
        tonnage_to_reassign = x_rebal[i_largest, j_overloaded]

        # Find second-best same-side station for this district
        district_side = districts_yaka[i_largest]
        same_side_stations = [j for j in range(N_STATIONS)
                             if stations_yaka[j] == district_side and j != j_overloaded]

        if not same_side_stations:
            # No alternative same-side station available
            break

        # Choose station with lowest distance among same-side alternatives
        alternative_distances = [(D_penalized[i_largest, j], j) for j in same_side_stations]
        alternative_distances.sort()
        j_alternative = alternative_distances[0][1]

        # Reassign
        x_rebal[i_largest, j_overloaded] = 0.0
        x_rebal[i_largest, j_alternative] = tonnage_to_reassign

    final_loads = np.sum(x_rebal, axis=0)
    final_overflow = np.sum(np.maximum(0.0, final_loads - Q_j))

    return x_rebal


def crossing_aware_pruning(x: np.ndarray,
                          t: np.ndarray,
                          d_i: np.ndarray,
                          districts_yaka: np.ndarray,
                          stations_yaka: np.ndarray,
                          N_DISTRICTS: int,
                          N_STATIONS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prune to single route per district, eliminating all Bosphorus crossings.

    For each district, selects the station with highest allocation (argmax x_ij)
    among same-side stations only. Assigns all district waste to that station.

    This is the final pruning step that guarantees:
    - Exactly 1 active route per district (39 total)
    - Zero Bosphorus crossings
    - All waste transported (Σ_j x_ij = d_i preserved)

    Args:
        x: Allocation matrix (N_DISTRICTS, N_STATIONS)
        t: Departure times matrix (N_DISTRICTS, N_STATIONS)
        d_i: Daily waste per district (N_DISTRICTS,)
        districts_yaka: Array of district sides ('Asya' or 'Avrupa')
        stations_yaka: Array of station sides ('Asya' or 'Avrupa')
        N_DISTRICTS: Number of districts
        N_STATIONS: Number of stations

    Returns:
        Tuple of (x_clean, t_clean): Pruned allocation and time matrices
    """
    x_raw = x.reshape(N_DISTRICTS, N_STATIONS)
    t_raw = t.reshape(N_DISTRICTS, N_STATIONS)

    x_clean = np.zeros_like(x_raw)
    t_clean = np.zeros_like(t_raw)

    for i in range(N_DISTRICTS):
        # Get district side
        district_side = districts_yaka[i]

        # Score all routes by current allocation
        scores = x_raw[i, :].copy()

        # Zero out scores for wrong-side stations
        for j in range(N_STATIONS):
            if stations_yaka[j] != district_side:
                scores[j] = -np.inf

        # Edge case: no same-side stations available (shouldn't happen with Istanbul data)
        if np.all(np.isinf(scores)):
            # Fall back to allowing crossings
            scores = x_raw[i, :].copy()

        # Select best same-side station
        best_j = int(np.argmax(scores))

        # Assign all waste to this station
        x_clean[i, best_j] = d_i[i]
        t_clean[i, best_j] = t_raw[i, best_j]

    return x_clean, t_clean
