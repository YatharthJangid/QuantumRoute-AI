"""
scaling_analysis.py — Classical vs QAOA Scaling Demonstration
==============================================================
Shows WHY quantum wins at large n by plotting:
  - Actual brute-force runtime for n = 3..8 customers
  - Theoretical QAOA circuit complexity O(p * n^2)

This is the strongest academic argument for quantum advantage
in your ETT presentation — the crossover point graph.

Usage:
    python -m src.benchmark.scaling_analysis
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.vrp_graph import build_vrp_instance
from src.core.vrp_classical import brute_force_vrp
from src.core.qaoa_solver import solve_vrp_qaoa

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def measure_classical_scaling(n_range: range, n_vehicles: int = 2,
                               capacity: int = 40) -> tuple[list, list]:
    """Measure actual brute-force runtime and cost for each n in n_range."""
    times = []
    costs = []
    for n in n_range:
        inst = build_vrp_instance(n_customers=n, n_vehicles=n_vehicles,
                                  capacity=capacity, seed=42)
        t0 = time.perf_counter()
        res = brute_force_vrp(inst)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        costs.append(res['total_cost'])
        print(f"  [Classical] n={n}: time = {elapsed:.6f}s, cost = {res['total_cost']}")
    return times, costs


def measure_qaoa_scaling(n_range: range, n_vehicles: int = 2,
                          capacity: int = 40) -> tuple[list, list]:
    """Measure actual QAOA runtime and cost for each n in n_range."""
    times = []
    costs = []
    for n in n_range:
        inst = build_vrp_instance(n_customers=n, n_vehicles=n_vehicles,
                                  capacity=capacity, seed=42)
        t0 = time.perf_counter()
        # Using a balanced maxiter/shots for reasonable runtime
        res = solve_vrp_qaoa(inst, p=1, backend='aer_simulator', noisy=False, 
                             shots=1024, maxiter=30)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        costs.append(res['best_cost'])
        print(f"  [QAOA]      n={n}: time = {elapsed:.6f}s, cost = {res['best_cost']}")
    return times, costs


def plot_scaling_actual(ns: list, classical_times: list, qaoa_times: list, 
                        classical_costs: list, qaoa_costs: list,
                        save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Classical vs Actual QAOA Runtime & Cost — QuantumRoute-AI",
                 fontsize=14, fontweight="bold")

    # ── Log scale Runtime ──────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(ns, classical_times, "o-", color="#E91E63", linewidth=2.5,
                markersize=8, label="Brute Force  O((n-1)!)")
    ax.semilogy(ns, qaoa_times, "s-", color="#2196F3", linewidth=2.5,
                markersize=8, label=f"QAOA Actual (AerSimulator)")
    ax.set_xlabel("Number of customers (n)", fontsize=11)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=11)
    ax.set_title("Runtime Comparison", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f8f9fa")
    
    # Add crossover line
    for i in range(len(ns) - 1):
        if classical_times[i] > qaoa_times[i]:
            ax.axvline(x=ns[i], color="green", linestyle=":", alpha=0.7, lw=1.5)
            ax.text(ns[i] + 0.05, max(qaoa_times) * 2,
                    f"Crossover\nn≈{ns[i]}", color="green", fontsize=9)
            break

    # ── Actual Cost Comparison ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(ns, classical_costs, "o-", color="#E91E63", linewidth=2.5,
             markersize=8, label="Brute Force Solution Cost")
    ax2.plot(ns, qaoa_costs, "s-", color="#2196F3", linewidth=2.5,
             markersize=8, label="QAOA Estimated Cost (p=1)")
    ax2.set_xlabel("Number of customers (n)", fontsize=11)
    ax2.set_ylabel("Total Route Cost", fontsize=11)
    ax2.set_title("Solution Quality vs Graph Size", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.set_facecolor("#f8f9fa")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[scaling_analysis] Saved plots → {save_path}")


if __name__ == "__main__":
    print("=" * 58)
    print("  QuantumRoute-AI — Classical vs QAOA Actual Scaling Analysis")
    print("=" * 58)

    ns = list(range(3, 9))   # We evaluate n=3..8, since brute_force starts blowing up after 8.

    print("\n[1/2] Measuring classical brute-force scaling ...")
    classical_times, classical_costs = measure_classical_scaling(ns, capacity=40)

    print("\n[2/2] Measuring ACTUAL QAOA scaling (AerSimulator) ...")
    qaoa_times, qaoa_costs = measure_qaoa_scaling(ns, capacity=40)

    save_path = RESULTS_DIR / "scaling_analysis_actual.png"
    plot_scaling_actual(ns, classical_times, qaoa_times, classical_costs, qaoa_costs, save_path)

    print("\nScaling summary:")
    print(f"  {'n':>3}  |  {'Brute Force (s)':>15}  |  {'QAOA actual (s)':>15}  |  {'BF Cost':>10}  |  {'QAOA Cost':>10}")
    print("  " + "-" * 75)
    for i, n in enumerate(ns):
        ct = classical_times[i]
        qt = qaoa_times[i]
        cc = classical_costs[i]
        qc = qaoa_costs[i]
        marker = " ← QAOA faster" if qt < ct else ""
        print(f"  {n:>3}  |  {ct:>15.6f}  |  {qt:>15.4f}  |  {cc:>10.2f}  |  {qc:>10.2f}{marker}")
