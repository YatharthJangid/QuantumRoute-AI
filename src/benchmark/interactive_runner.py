"""
interactive_runner.py — Interactive single-instance runner
==========================================================
Run a custom VRP instance by answering interactive prompts.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.vrp_graph import build_vrp_instance, validate_vrp
from src.core.vrp_classical import brute_force_vrp, greedy_vrp
from src.core.qaoa_solver import solve_vrp_qaoa


def prompt_int(message: str, default: int) -> int:
    try:
        val = input(f"{message} [default {default}]: ").strip()
        return int(val) if val else default
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default


def prompt_bool(message: str, default: bool) -> bool:
    def_str = "Y/n" if default else "y/N"
    val = input(f"{message} ({def_str}): ").strip().lower()
    if not val:
        return default
    return val in ['y', 'yes', 'true', '1']


def main():
    print("=" * 60)
    print("  QuantumRoute-AI — Interactive Benchmark Runner")
    print("=" * 60)
    print("Enter parameters for your VRP run, or press Enter to keep defaults.\n")

    # 1. Gather parameters
    n_customers = prompt_int("Number of customers (n)", 5)
    n_vehicles = prompt_int("Number of vehicles", 2)
    capacity = prompt_int("Vehicle capacity", 30)

    # Note on classical brute force scaling (O((n-1)!))
    if n_customers > 9:
        print(f"\n⚠️  WARNING: n={n_customers} will take a VERY long time for Brute Force!")
        print("    Brute force scales as O((n-1)!).")
        print("    n=8 takes  ~0.2 s")
        print("    n=10 takes ~20 s")
        print("    n=11 takes ~3 minutes")
        print("    n=12 takes ~30 minutes\n")

    run_bf = prompt_bool("Run Brute Force (Exact Classical)?", default=(n_customers <= 9))
    run_greedy = prompt_bool("Run Greedy (Heuristic Classical)?", default=True)
    run_qaoa = prompt_bool("Run QAOA (Quantum Simulator)?", default=True)

    # QAOA params if requested
    qaoa_p = 1
    qaoa_maxiter = 30
    if run_qaoa:
        print("")
        qaoa_p = prompt_int("  QAOA Circuit Depth (p)", 1)
        qaoa_maxiter = prompt_int("  QAOA Max Iterations (COBYLA)", 30)

    # 2. Build Instance
    print("\n" + "─" * 60)
    print(f"Building VRP instance with {n_customers} customers...")
    instance = build_vrp_instance(
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        capacity=capacity,
        seed=42
    )
    validate_vrp(instance)
    print("─" * 60)

    # 3. Run selected solvers
    results = {}

    if run_greedy:
        print("\n⚙️  Running Greedy Nearest-Neighbour...")
        t0 = time.perf_counter()
        res_greedy = greedy_vrp(instance)
        elapsed = time.perf_counter() - t0
        print(f"  Cost: {res_greedy['total_cost']}")
        print(f"  Time: {elapsed:.6f}s")
        results['Greedy'] = (res_greedy['total_cost'], elapsed)

    if run_bf:
        print("\n⚙️  Running Classical Brute Force... (this might take a while)")
        t0 = time.perf_counter()
        res_bf = brute_force_vrp(instance)
        elapsed = time.perf_counter() - t0
        print(f"  Cost: {res_bf['total_cost']}")
        print(f"  Time: {elapsed:.6f}s")
        for veh, route in res_bf['routes'].items():
            print(f"    {veh}: {' -> '.join(map(str, route))}")
        results['Brute Force'] = (res_bf['total_cost'], elapsed)

    if run_qaoa:
        print(f"\n⚛️  Running QAOA Simulator (p={qaoa_p}, maxiter={qaoa_maxiter})...")
        print("     (Simulating quantum circuits... please wait)")
        t0 = time.perf_counter()
        res_qaoa = solve_vrp_qaoa(
            instance,
            p=qaoa_p,
            backend='aer_simulator',
            shots=1024,
            maxiter=qaoa_maxiter
        )
        elapsed = time.perf_counter() - t0
        # Print a newline to prevent overlap from qaoa output logs
        print(f"\n  Cost: {res_qaoa['best_cost']}")
        print(f"  Time: {elapsed:.6f}s")
        for veh, route in res_qaoa['routes'].items():
            print(f"    {veh}: {' -> '.join(map(str, route))}")
        results['QAOA'] = (res_qaoa['best_cost'], elapsed)

    # 4. Summary
    print("\n" + "=" * 60)
    print(f"  🏁 SUMMARY FOR n={n_customers} ")
    print("=" * 60)
    if results:
        for solver, (cost, t) in results.items():
            print(f"  {solver:<15} |  Cost: {cost:<8.2f} |  Runtime: {t:.4f}s")
    else:
        print("  No solvers were run.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
