import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")
COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_route_map(
    coords: list,
    routes: list,
    title: str = "VRP Route Map",
    save_path: str = None,
):
    """
    Plot city nodes and vehicle routes on a 2D map.

    Args:
        coords:  list of (x, y) tuples, index 0 = depot
        routes:  list of routes, each route = list of node indices
                 e.g. [[0,1,2,0], [0,3,4,0]]
        title:   plot title
        save_path: if None, saves to results/route_map.png
    """
    _ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "route_map.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Draw routes
    legend_patches = []
    for i, route in enumerate(routes):
        color = COLORS[i % len(COLORS)]
        for j in range(len(route) - 1):
            x_vals = [coords[route[j]][0], coords[route[j + 1]][0]]
            y_vals = [coords[route[j]][1], coords[route[j + 1]][1]]
            ax.plot(x_vals, y_vals, color=color, linewidth=2,
                    alpha=0.8, zorder=1)
            # Arrow midpoint
            mx = (x_vals[0] + x_vals[1]) / 2
            my = (y_vals[0] + y_vals[1]) / 2
            dx = x_vals[1] - x_vals[0]
            dy = y_vals[1] - y_vals[0]
            ax.annotate("", xy=(mx + dx * 0.01, my + dy * 0.01),
                        xytext=(mx - dx * 0.01, my - dy * 0.01),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.5))
        patch = mpatches.Patch(color=color, label=f"Vehicle {i + 1}")
        legend_patches.append(patch)

    # Draw depot
    ax.scatter(xs[0], ys[0], c="#FF5722", s=200, zorder=5,
               marker="*", label="Depot")
    ax.annotate("Depot", (xs[0], ys[0]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, fontweight="bold", color="#FF5722")

    # Draw cities
    ax.scatter(xs[1:], ys[1:], c="#37474F", s=80, zorder=4)
    for i in range(1, len(coords)):
        ax.annotate(f"C{i}", (xs[i], ys[i]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="#37474F")

    ax.legend(handles=legend_patches + [
        mpatches.Patch(color="#FF5722", label="Depot")
    ], loc="upper left", framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Route map saved -> {save_path}")
    return save_path


def plot_performance_comparison(
    results: list,
    title: str = "Solver Performance Comparison",
    save_path: str = None,
):
    """
    Bar chart comparing solvers on cost and runtime.

    Args:
        results: list of dicts with keys:
                 'solver', 'best_cost', 'runtime_s'
        save_path: if None, saves to results/performance_comparison.png
    """
    _ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "performance_comparison.png")

    solvers = [r["solver"] for r in results]
    costs = [float(r["best_cost"]) for r in results]
    runtimes = [float(r["runtime_s"]) for r in results]
    x = np.arange(len(solvers))
    bar_w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # Cost chart
    bars1 = ax1.bar(x, costs, width=bar_w * 2,
                    color=COLORS[:len(solvers)], alpha=0.85, edgecolor="white")
    ax1.set_title("Route Cost (lower is better)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(solvers, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Total Cost")
    ax1.set_facecolor("#f8f9fa")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars1, costs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(costs) * 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    # Runtime chart
    bars2 = ax2.bar(x, runtimes, width=bar_w * 2,
                    color=COLORS[:len(solvers)], alpha=0.85, edgecolor="white")
    ax2.set_title("Runtime in seconds (lower is better)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(solvers, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Runtime (s)")
    ax2.set_facecolor("#f8f9fa")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars2, runtimes):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(runtimes) * 0.01,
                 f"{val:.3f}s", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Performance chart saved -> {save_path}")
    return save_path


if __name__ == "__main__":
    coords = [
        (5, 5),   # depot
        (1, 3), (2, 8), (6, 9),
        (9, 2), (8, 7),
    ]
    routes = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]
    plot_route_map(coords, routes, title="QAOA VRP Solution — 2 Vehicles")

    sample_results = [
        {"solver": "brute_force",  "best_cost": 18.5,  "runtime_s": 0.032},
        {"solver": "qaoa_ideal",   "best_cost": 19.2,  "runtime_s": 4.81},
        {"solver": "qaoa_noisy",   "best_cost": 21.1,  "runtime_s": 5.34},
    ]
    plot_performance_comparison(sample_results)
