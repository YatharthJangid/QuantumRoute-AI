import csv
import os
from datetime import datetime


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")
CSV_FIELDS = [
    "timestamp", "solver", "num_vehicles", "num_cities",
    "best_cost", "runtime_s", "backend", "routes", "noise_deviation_pct"
]


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def export_result(
    solver: str,
    num_vehicles: int,
    num_cities: int,
    best_cost: float,
    runtime_s: float,
    backend: str = "aer_simulator",
    routes: list = None,
    ideal_cost: float = None,
):
    """
    Append a single solver result row to results/vrp_results.csv.

    Args:
        solver:           e.g. 'brute_force', 'qaoa_ideal', 'qaoa_noisy'
        num_vehicles:     number of vehicles used
        num_cities:       number of cities (excluding depot)
        best_cost:        total route cost found
        runtime_s:        wall-clock runtime in seconds
        backend:          Qiskit backend name or 'classical'
        routes:           list of route lists, e.g. [[0,1,2,0], [0,3,4,0]]
        ideal_cost:       if provided, compute noise deviation % vs this baseline
    """
    ensure_results_dir()
    csv_path = os.path.join(RESULTS_DIR, "vrp_results.csv")
    write_header = not os.path.exists(csv_path)

    noise_deviation_pct = ""
    if ideal_cost is not None and ideal_cost > 0:
        noise_deviation_pct = round(
            abs(best_cost - ideal_cost) / ideal_cost * 100, 2
        )

    routes_str = str(routes) if routes else ""

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "solver": solver,
            "num_vehicles": num_vehicles,
            "num_cities": num_cities,
            "best_cost": round(best_cost, 4),
            "runtime_s": round(runtime_s, 4),
            "backend": backend,
            "routes": routes_str,
            "noise_deviation_pct": noise_deviation_pct,
        })

    print(f"[results_exporter] Saved: {solver} | cost={best_cost:.4f} | "
          f"runtime={runtime_s:.4f}s -> {csv_path}")
    return csv_path


def analyze_noise(ideal_cost: float, noisy_cost: float) -> dict:
    """
    Compare ideal QAOA vs noisy QAOA result and return analysis dict.
    """
    if ideal_cost <= 0:
        return {"error": "ideal_cost must be > 0"}

    deviation = abs(noisy_cost - ideal_cost)
    deviation_pct = (deviation / ideal_cost) * 100

    return {
        "ideal_cost": round(ideal_cost, 4),
        "noisy_cost": round(noisy_cost, 4),
        "absolute_deviation": round(deviation, 4),
        "noise_deviation_pct": round(deviation_pct, 2),
        "verdict": "acceptable" if deviation_pct < 10 else "significant noise",
    }


def load_results(csv_path: str = None) -> list:
    """Load all rows from vrp_results.csv as a list of dicts."""
    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, "vrp_results.csv")
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    export_result(
        solver="brute_force",
        num_vehicles=2,
        num_cities=4,
        best_cost=18.5,
        runtime_s=0.032,
        backend="classical",
        routes=[[0, 1, 2, 0], [0, 3, 4, 0]],
    )
    export_result(
        solver="qaoa_ideal",
        num_vehicles=2,
        num_cities=4,
        best_cost=19.2,
        runtime_s=4.81,
        backend="aer_simulator",
        routes=[[0, 1, 2, 0], [0, 3, 4, 0]],
    )
    export_result(
        solver="qaoa_noisy",
        num_vehicles=2,
        num_cities=4,
        best_cost=21.1,
        runtime_s=5.34,
        backend="ibm_brisbane",
        routes=[[0, 1, 3, 0], [0, 2, 4, 0]],
        ideal_cost=19.2,
    )

    noise = analyze_noise(ideal_cost=19.2, noisy_cost=21.1)
    print("\nNoise Analysis:", noise)

    rows = load_results()
    print(f"\nTotal results logged: {len(rows)}")
