"""
Visualization functions for Istanbul waste routing optimization.

REQ-V2: Station Load Bar Chart
REQ-V3: Departure Time Histogram

Usage in app.ipynb (run AFTER the slider cell):
    from src.plot_results import plot_station_loads, plot_departure_times
    plot_station_loads(x_opt, Q_j, station_names)
    plot_departure_times(x_opt, t_opt, smooth_traffic_curve)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_station_loads(x_opt: np.ndarray,
                       Q_j: np.ndarray,
                       station_names: list,
                       save_path: str = "outputs/station_loads.png") -> None:
    """
    REQ-V2: Bar chart comparing each station's capacity vs. actual load.

    Color coding:
      - Green  : utilization < 90%  (comfortable)
      - Orange : utilization 90-110% (near capacity)
      - Red    : utilization > 110%  (overflow)

    Args:
        x_opt        : Allocation matrix (N_DISTRICTS, N_STATIONS) after pruning
        Q_j          : Station capacities array (N_STATIONS,)
        station_names: List of full station name strings
        save_path    : Output file path for the PNG
    """
    station_loads = np.sum(x_opt, axis=0)
    short_names   = [n.split()[0] for n in station_names]
    n_st          = len(short_names)
    x_pos         = np.arange(n_st)
    bar_width     = 0.35

    def load_color(load, cap):
        pct = load / cap
        if pct > 1.10:
            return '#e74c3c'   # red   — overflow
        elif pct > 0.90:
            return '#f39c12'   # orange — near capacity
        return '#2ecc71'       # green  — comfortable

    colors = [load_color(station_loads[j], Q_j[j]) for j in range(n_st)]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Capacity bars (grey, side-by-side left)
    ax.bar(x_pos - bar_width / 2, Q_j, bar_width,
           color='#4a4a6a', label='Capacity (ton/day)', zorder=2)

    # Load bars (colored, side-by-side right)
    ax.bar(x_pos + bar_width / 2, station_loads, bar_width,
           color=colors, label='Actual Load (ton/day)', zorder=2)

    # Percentage labels above each load bar
    for j in range(n_st):
        pct = station_loads[j] / Q_j[j] * 100
        ax.text(x_pos[j] + bar_width / 2, station_loads[j] + 15,
                f'{pct:.0f}%', ha='center', va='bottom',
                color='white', fontsize=8, fontweight='bold')

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=30, ha='right',
                       color='#cccccc', fontsize=9)
    ax.set_ylabel('Waste (ton/day)', color='#cccccc')
    ax.set_title('Station Capacity vs. Actual Load — REQ-V2',
                 color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#cccccc')
    ax.spines[:].set_color('#333355')
    ax.yaxis.label.set_color('#cccccc')
    ax.grid(axis='y', color='#333355', linestyle='--', alpha=0.5, zorder=1)

    # Legend
    red_patch    = mpatches.Patch(color='#e74c3c', label='>110% (overflow)')
    orange_patch = mpatches.Patch(color='#f39c12', label='90–110% (near capacity)')
    green_patch  = mpatches.Patch(color='#2ecc71', label='<90% (ok)')
    grey_patch   = mpatches.Patch(color='#4a4a6a', label='Capacity')
    ax.legend(handles=[grey_patch, green_patch, orange_patch, red_patch],
              facecolor='#1a1a2e', edgecolor='#444466',
              labelcolor='white', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()

    total_overflow = float(np.sum(np.maximum(0, station_loads - Q_j)))
    print(f"✅ Saved to {save_path}")
    print(f"   Total overflow: {total_overflow:.1f} ton/day")


def plot_departure_times(x_opt: np.ndarray,
                         t_opt: np.ndarray,
                         smooth_traffic_curve,
                         save_path: str = "outputs/departure_times.png") -> None:
    """
    REQ-V3: Histogram of optimized departure times overlaid on τ(t) congestion curve.

    Left axis : histogram of t_ij for active routes (x_ij > 0.5 ton/day)
    Right axis: τ(t) = 1 - (v(t) - v_min) / (v_max - v_min) congestion index

    If the optimizer correctly avoids peak hours, the histogram should show
    fewer departures at 07-09 and 17-19 compared to off-peak hours.

    Args:
        x_opt               : Allocation matrix (N_DISTRICTS, N_STATIONS)
        t_opt               : Departure times matrix (N_DISTRICTS, N_STATIONS)
        smooth_traffic_curve: CubicSpline object returning v(t) in km/h
        save_path           : Output file path for the PNG
    """
    # Departure times for active routes only
    active_mask     = x_opt > 0.5
    departure_times = t_opt[active_mask].flatten()

    # τ(t) congestion curve on fine grid
    t_grid   = np.linspace(6, 22, 300)
    v_grid   = smooth_traffic_curve(t_grid)
    tau_grid = np.clip(1.0 - (v_grid - 27.0) / 9.0, 0.0, 1.0)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#16213e')

    # Highlight AM and PM peak hour bands
    for start, end in [(7, 9), (17, 19)]:
        ax1.axvspan(start, end, alpha=0.12, color='#e74c3c', zorder=1)

    # Left axis: departure histogram (hourly bins, 6 to 22)
    bins = np.arange(6, 23, 1)
    ax1.hist(departure_times, bins=bins, color='#3498db',
             edgecolor='#1a1a2e', alpha=0.80, zorder=2, label='Active departures')
    ax1.set_xlabel('Departure Hour', color='#cccccc', fontsize=10)
    ax1.set_ylabel('Number of Active Routes', color='#3498db', fontsize=10)
    ax1.tick_params(axis='y', colors='#3498db')
    ax1.tick_params(axis='x', colors='#cccccc')
    ax1.set_xlim(6, 22)
    ax1.spines[:].set_color('#333355')
    ax1.grid(axis='y', color='#333355', linestyle='--', alpha=0.4, zorder=0)

    # Right axis: τ(t) congestion curve
    ax2 = ax1.twinx()
    ax2.plot(t_grid, tau_grid, color='#e74c3c', linewidth=2.5,
             linestyle='--', label='τ(t) congestion', zorder=3)
    ax2.set_ylabel('τ(t) — Congestion Index (0=free, 1=peak)',
                   color='#e74c3c', fontsize=10)
    ax2.tick_params(colors='#e74c3c')
    ax2.set_ylim(0, 1.3)
    ax2.spines[:].set_color('#333355')

    # Peak zone labels
    y_top = ax1.get_ylim()[1]
    ax1.text(8,  y_top * 0.92, 'AM\nPeak', ha='center', va='top',
             color='#e74c3c', fontsize=8, fontweight='bold')
    ax1.text(18, y_top * 0.92, 'PM\nPeak', ha='center', va='top',
             color='#e74c3c', fontsize=8, fontweight='bold')

    # Combined legend from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor='#1a1a2e', edgecolor='#444466',
               labelcolor='white', fontsize=9, loc='upper left')

    ax1.set_title('Optimized Departure Times vs. Traffic Congestion — REQ-V3',
                  color='white', fontsize=12, pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()

    print(f"✅ Saved to {save_path}")
    print(f"   Active routes analysed : {len(departure_times)}")
    print(f"   Mean departure time    : {departure_times.mean():.2f}h")
    print(f"   Std departure time     : {departure_times.std():.2f}h")
