"""
interactive_runner.py - Interactive single-instance runner
==========================================================
Run a custom VRP instance by answering interactive prompts.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.vrp_graph import analyze_vrp_instance, build_vrp_instance, validate_vrp
from src.core.vrp_classical import brute_force_vrp, clarke_wright_vrp, greedy_vrp
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
    return val in ["y", "yes", "true", "1"]


def _print_solver_result(name: str, result: dict) -> None:
    print(f"  Cost    : {result.get('best_cost', result.get('total_cost'))}")
    print(f"  Time    : {result.get('runtime_s', '?')}s")
    if "feasible" in result:
        print(f"  Feasible: {result['feasible']}")
    if result.get("status"):
        print(f"  Status  : {result['status']}")
    for veh, route in result.get("routes", {}).items():
        print(f"    {veh}: {' -> '.join(map(str, route))}")
    if result.get("unserved"):
        print(f"    Unserved: {result['unserved']}")


def main():
    print("=" * 60)
    print("  QuantumRoute-AI - Interactive Benchmark Runner")
    print("=" * 60)
    print("Enter parameters for your VRP run, or press Enter to keep defaults.\n")

    n_customers = prompt_int("Number of customers (n)", 5)
    n_vehicles = prompt_int("Number of vehicles", 2)
    capacity = prompt_int("Vehicle capacity", 30)

    if n_customers > 9:
        print(f"\nWARNING: n={n_customers} will take a very long time for brute force.")
        print("  Brute force scales as O((n-1)!).")
        print("  n=8 takes  ~0.2 s")
        print("  n=10 takes ~20 s")
        print("  n=11 takes ~3 minutes")
        print("  n=12 takes ~30 minutes\n")

    run_bf = prompt_bool("Run Brute Force (Exact Classical)?", default=(n_customers <= 9))
    run_greedy = prompt_bool("Run Greedy (Heuristic Classical)?", default=True)
    run_clarke_wright = prompt_bool("Run Clarke-Wright Savings?", default=True)
    qaoa_default = n_vehicles == 2
    run_qaoa = prompt_bool(
        "Run QAOA (Quantum Simulator, 2 vehicles only)?",
        default=qaoa_default,
    )

    qaoa_p = 1
    qaoa_maxiter = 30
    if run_qaoa:
        print("")
        qaoa_p = prompt_int("  QAOA Circuit Depth (p)", 1)
        qaoa_maxiter = prompt_int("  QAOA Max Iterations (COBYLA)", 30)

    print("\n" + "-" * 60)
    print(f"Building VRP instance with {n_customers} customers...")
    instance = build_vrp_instance(
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        capacity=capacity,
        seed=42,
    )
    analysis = validate_vrp(instance)
    print("-" * 60)

    results = {}

    if run_greedy:
        print("\nRunning Greedy Nearest-Neighbour...")
        res_greedy = greedy_vrp(instance)
        _print_solver_result("Greedy", res_greedy)
        results["Greedy"] = res_greedy.get("total_cost")

    if run_clarke_wright:
        print("\nRunning Clarke-Wright Savings...")
        res_cw = clarke_wright_vrp(instance)
        _print_solver_result("Clarke-Wright", res_cw)
        results["Clarke-Wright"] = res_cw.get("total_cost")

    if run_bf:
        print("\nRunning Classical Brute Force... (this might take a while)")
        res_bf = brute_force_vrp(instance)
        _print_solver_result("Brute Force", res_bf)
        results["Brute Force"] = res_bf.get("total_cost")

    if run_qaoa:
        if n_vehicles != 2:
            print("\nSkipping QAOA: the current implementation supports exactly 2 vehicles.")
        elif not analysis["is_feasible"]:
            print(
                "\nSkipping QAOA: the instance is infeasible for the declared fleet. "
                + "; ".join(analysis["infeasible_reasons"])
            )
        else:
            print(f"\nRunning QAOA Simulator (p={qaoa_p}, maxiter={qaoa_maxiter})...")
            print("  Simulating quantum circuits... please wait")
            res_qaoa = solve_vrp_qaoa(
                instance,
                p=qaoa_p,
                backend="aer_simulator",
                shots=1024,
                maxiter=qaoa_maxiter,
            )
            _print_solver_result("QAOA", res_qaoa)
            results["QAOA"] = res_qaoa.get("best_cost")

    print("\n" + "=" * 60)
    print(f"  SUMMARY FOR n={n_customers}")
    print("=" * 60)
    if results:
        for solver, cost in results.items():
            print(f"  {solver:<15} | Cost: {cost}")
    else:
        print("  No solvers were run.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
