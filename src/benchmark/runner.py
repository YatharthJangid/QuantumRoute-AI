"""
runner.py - QuantumRoute-AI Benchmark Runner
============================================
Runs the available VRP solvers on the same VRPInstance and collects results:
    1. brute_force      (exact, classical)
    2. greedy           (heuristic, classical)
    3. clarke_wright    (heuristic, classical)
    4. qaoa_ideal       (QAOA with ideal AerSimulator, 2 vehicles only)
    5. qaoa_noisy       (QAOA with depolarising noise model, 2 vehicles only)

Each result is exported to results/vrp_results.csv via results_exporter,
and the run ends with a performance comparison chart saved to
results/performance_comparison.png.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.vrp_graph import VRPInstance, analyze_vrp_instance, build_vrp_instance
from src.core.vrp_classical import brute_force_vrp, clarke_wright_vrp, greedy_vrp
from src.core.qaoa_solver import solve_vrp_qaoa
from src.core.results_exporter import analyze_noise, export_result
from src.core.visualizer import plot_performance_comparison, plot_route_map


def _routes_to_list(routes_dict: dict) -> list:
    return list(routes_dict.values())


def _print_section(title: str) -> None:
    bar = "-" * 58
    print(f"\n+{bar}+")
    print(f"|  {title:<56}|")
    print(f"+{bar}+")


def _print_result(solver_name: str, result: dict, instance: VRPInstance) -> None:
    G = instance.graph
    depot = instance.depot
    print(f"  Solver  : {solver_name}")
    for veh, route in result.get("routes", {}).items():
        cost = sum(
            G[route[i]][route[i + 1]].get("weight", 0)
            for i in range(len(route) - 1)
            if route[i] != route[i + 1]
        )
        load = sum(instance.demands[n] for n in route if n != depot)
        print(f"    {veh}: {' -> '.join(map(str, route))}  [cost={cost:.2f}, load={load}/{instance.capacity}]")
    print(f"  Total cost : {result.get('best_cost', result.get('total_cost', '?'))}")
    print(f"  Runtime    : {result.get('runtime_s', '?')}s")
    if "feasible" in result:
        print(f"  Feasible   : {result['feasible']}")
    if result.get("status"):
        print(f"  Status     : {result['status']}")
    if result.get("unserved"):
        print(f"  Unserved   : {result['unserved']}")


def _record_result(
    all_results: dict,
    comparison_data: list,
    solver: str,
    result: dict,
    num_vehicles: int,
    num_cities: int,
    backend: str,
    ideal_cost: float | None = None,
) -> None:
    best_cost = result.get("best_cost", result.get("total_cost"))
    runtime_s = result.get("runtime_s", 0.0)
    export_result(
        solver=solver,
        num_vehicles=num_vehicles,
        num_cities=num_cities,
        best_cost=best_cost,
        runtime_s=runtime_s,
        backend=backend,
        routes=_routes_to_list(result.get("routes", {})),
        ideal_cost=ideal_cost,
    )
    all_results[solver] = {**result, "best_cost": best_cost, "runtime_s": runtime_s}
    comparison_data.append({
        "solver": solver,
        "best_cost": best_cost,
        "runtime_s": runtime_s,
    })


def run_benchmark(
    num_cities: int = 4,
    num_vehicles: int = 2,
    capacity: int = 10,
    seed: int = 42,
    qaoa_p: int = 1,
    qaoa_shots: int = 1024,
    qaoa_maxiter: int = 60,
    noise_rate: float = 0.01,
    run_qaoa: bool = True,
) -> dict:
    """
    Run every supported VRP solver on the same instance and export results.

    Returns a dict keyed by solver name.
    """
    print("\n" + "=" * 60)
    print("  QuantumRoute-AI - Full Benchmark Runner")
    print("=" * 60)
    print(f"  Config: cities={num_cities}, vehicles={num_vehicles}, capacity={capacity}, seed={seed}")
    print(f"  QAOA  : p={qaoa_p}, shots={qaoa_shots}, maxiter={qaoa_maxiter}")
    print(f"  Noise : rate={noise_rate}")

    instance = build_vrp_instance(
        n_customers=num_cities,
        n_vehicles=num_vehicles,
        capacity=capacity,
        seed=seed,
    )
    analysis = analyze_vrp_instance(instance)
    coords = list(instance.positions.values())

    all_results: dict = {}
    comparison_data: list = []

    _print_section("1 / 5  Brute-Force (Classical)")
    bf_result = brute_force_vrp(instance)
    _print_result("brute_force", bf_result, instance)
    _record_result(
        all_results,
        comparison_data,
        solver="brute_force",
        result=bf_result,
        num_vehicles=num_vehicles,
        num_cities=num_cities,
        backend="classical",
    )

    _print_section("2 / 5  Greedy Nearest-Neighbour (Classical)")
    greedy_result = greedy_vrp(instance)
    _print_result("greedy", greedy_result, instance)
    _record_result(
        all_results,
        comparison_data,
        solver="greedy",
        result=greedy_result,
        num_vehicles=num_vehicles,
        num_cities=num_cities,
        backend="classical",
    )

    _print_section("3 / 5  Clarke-Wright Savings (Classical)")
    cw_result = clarke_wright_vrp(instance)
    _print_result("clarke_wright", cw_result, instance)
    _record_result(
        all_results,
        comparison_data,
        solver="clarke_wright",
        result=cw_result,
        num_vehicles=num_vehicles,
        num_cities=num_cities,
        backend="classical",
    )

    if not run_qaoa:
        print("\n[runner] Skipping QAOA solvers (run_qaoa=False).")
    elif num_vehicles != 2:
        print(
            "\n[runner] Skipping QAOA solvers because the current implementation "
            f"supports exactly 2 vehicles, not {num_vehicles}."
        )
    elif not analysis["is_feasible"]:
        print(
            "\n[runner] Skipping QAOA solvers because the instance is infeasible: "
            + "; ".join(analysis["infeasible_reasons"])
        )
    else:
        _print_section("4 / 5  QAOA - Ideal Simulator")
        qi_result = solve_vrp_qaoa(
            instance,
            p=qaoa_p,
            backend="aer_simulator",
            noisy=False,
            shots=qaoa_shots,
            maxiter=qaoa_maxiter,
            seed=seed,
        )
        _print_result("qaoa_ideal", qi_result, instance)
        _record_result(
            all_results,
            comparison_data,
            solver="qaoa_ideal",
            result=qi_result,
            num_vehicles=num_vehicles,
            num_cities=num_cities,
            backend="aer_simulator",
            ideal_cost=bf_result.get("total_cost"),
        )

        _print_section("5 / 5  QAOA - Noisy Simulator")
        qn_result = solve_vrp_qaoa(
            instance,
            p=qaoa_p,
            backend="aer_simulator",
            noisy=True,
            error_rate=noise_rate,
            shots=qaoa_shots,
            maxiter=qaoa_maxiter,
            seed=seed,
        )
        _print_result("qaoa_noisy", qn_result, instance)
        _record_result(
            all_results,
            comparison_data,
            solver="qaoa_noisy",
            result=qn_result,
            num_vehicles=num_vehicles,
            num_cities=num_cities,
            backend="aer_simulator_noisy",
            ideal_cost=qi_result["best_cost"],
        )

        noise_stats = analyze_noise(ideal_cost=qi_result["best_cost"], noisy_cost=qn_result["best_cost"])
        print(f"\n[noise analysis] {noise_stats}")

    _print_section("Generating Visualisations")
    bf_routes_list = _routes_to_list(bf_result["routes"])
    plot_route_map(
        coords=coords,
        routes=bf_routes_list,
        title=f"Brute Force VRP Routes - {num_cities} customers, {num_vehicles} vehicles",
    )
    plot_performance_comparison(
        results=comparison_data,
        title=f"Solver Comparison - {num_cities} customers, {num_vehicles} vehicles",
    )

    print("\n[runner] Benchmark complete. Results saved to results/vrp_results.csv")
    print("[runner] Charts saved to results/")
    return all_results


if __name__ == "__main__":
    results = run_benchmark(
        num_cities=4,
        num_vehicles=2,
        capacity=10,
        seed=42,
        qaoa_p=1,
        qaoa_shots=1024,
        qaoa_maxiter=60,
        noise_rate=0.01,
        run_qaoa=True,
    )

    print("\n" + "=" * 30)
    print("SUMMARY")
    print("=" * 30)
    for solver, res in results.items():
        cost = res.get("best_cost", res.get("total_cost", "?"))
        runtime = res.get("runtime_s", "?")
        print(f"  {solver:<15} cost={cost:<10} runtime={runtime}s")
