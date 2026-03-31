"""
qaoa_solver.py — Hybrid Quantum-Classical VRP Solver (Qiskit 1.x)
===================================================================
Encodes the 2-vehicle VRP as a QUBO where binary variable x_i = 1
means customer i is assigned to Vehicle-1, x_i = 0 → Vehicle-2.

QUBO objective (minimise):
    H = Σ_{i<j} w_ij * (x_i XOR x_j encoded as cost)
      + penalty for capacity violations per vehicle

The QAOA circuit (depth p=1) is built manually using Qiskit 1.x
QuantumCircuit, then sampled via AerSimulator (ideal or noisy).

Public API
----------
    solve_vrp_qaoa(vrp_instance, p=1, backend='aer_simulator') -> dict
        Returns:
            {
                'routes'    : {'vehicle_1': [...], 'vehicle_2': [...]},
                'best_cost' : float,
                'runtime_s' : float,
                'backend'   : str,
                'method'    : 'qaoa_ideal' | 'qaoa_noisy',
            }
"""

import sys
import time
from pathlib import Path

import numpy as np

# ── Qiskit 1.x imports ───────────────────────────────────────────────────────
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.vrp_graph import VRPInstance, build_vrp_instance

# ──────────────────────────────────────────────────────────────────────────────
# 1.  QUBO Construction
# ──────────────────────────────────────────────────────────────────────────────

def _build_qubo(instance: VRPInstance) -> tuple[np.ndarray, list[int]]:
    """
    Encode VRP (2 vehicles) as an upper-triangular QUBO matrix Q.

    Binary variable x_i = 1  →  customer i served by Vehicle-1
                      x_i = 0  →  customer i served by Vehicle-2

    Objective = travel cost (from depot, through assigned customers, back) +
                capacity-violation penalty.

    Returns
    -------
    Q        : (n, n) float numpy array (upper-triangular QUBO coefficients)
    customers: list of customer node IDs in variable order
    """
    G = instance.graph
    depot = instance.depot
    customers = [n for n in sorted(G.nodes()) if n != depot]
    n = len(customers)

    Q = np.zeros((n, n))

    # Edge weights between depot and customers
    w_depot = np.array(
        [G[depot][c].get("weight", 0) for c in customers], dtype=float
    )

    # Pairwise edge weights among customers
    W = np.zeros((n, n))
    for i, ci in enumerate(customers):
        for j, cj in enumerate(customers):
            if i != j:
                W[i, j] = G[ci][cj].get("weight", 0) if G.has_edge(ci, cj) else 0

    # --- Travel-cost contribution ---
    # Route cost for vehicle 1 (x_i = 1 set):
    #   C_v1 ≈ 2 * Σ_i  w_depot[i] * x_i  +  Σ_{i<j} W[i,j] * x_i * x_j
    # (rough linearisation; exact only for single-vehicle TSP, but captures the
    #  shape of the cost landscape correctly for QAOA demo purposes)
    for i in range(n):
        Q[i, i] += 2.0 * w_depot[i]          # depot-to-customer round trip
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += W[i, j]               # inter-customer edge

    # Route cost for vehicle 2 (x_i = 0 set, i.e. (1-x_i) = 1):
    # Expand and add to diagonal; cross-terms also added.
    for i in range(n):
        Q[i, i] += 2.0 * w_depot[i] - 2.0 * w_depot[i]  # self-cancels for depot
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] -= W[i, j]               # subtract; x=0 for vehicle 2

    # Rebuild correctly: cost(v1) + cost(v2) in QUBO form
    # Using the proper formulation:
    # cost = Σ_i w_depot[i]*(2*x_i*(1-x_i) + 2*(1-x_i)*x_i)  — always 0
    # Better: penalise BAD assignments via capacity constraints.
    # Reset and use penalty-driven QUBO (standard for VRP QUBO in literature)
    Q = np.zeros((n, n))

    capacity = instance.capacity
    penalty = float(max(w_depot) * n * 2)   # large enough to dominate

    demands = np.array([instance.demands[c] for c in customers], dtype=float)

    # Capacity penalty for Vehicle-1: (Σ x_i * d_i - C)^2 → penalise if > C
    # We use: pen * (Σ d_i*x_i - C)^2  expanded quadratically
    for i in range(n):
        Q[i, i] += penalty * (demands[i] ** 2 - 2 * capacity * demands[i])
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += penalty * 2 * demands[i] * demands[j]

    # Capacity penalty for Vehicle-2: (Σ (1-x_i)*d_i - C)^2
    # Expand: Σ d_i^2(1-x_i)^2 - 2C*Σ d_i*(1-x_i) + C^2  (constant dropped)
    for i in range(n):
        # (1-x_i)^2 = 1 - 2x_i + x_i^2;  treat x_i^2 = x_i for binary
        Q[i, i] += penalty * (demands[i] ** 2 * (-2 + 1) + 2 * capacity * demands[i]
                               - demands[i] ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            # cross term: 2 * d_i * d_j * (1-x_i)(1-x_j)  → contributes +2d_id_j to diagonal offset
            Q[i, j] += penalty * 2 * demands[i] * demands[j]

    # Travel-cost objective (linearised euclidean preference)
    for i in range(n):
        Q[i, i] += w_depot[i]                 # prefer nearby customers in v1
    for i in range(n):
        for j in range(i + 1, n):
            # Reward putting nearby customers in the same vehicle
            Q[i, j] -= 0.5 * W[i, j]

    return Q, customers


# ──────────────────────────────────────────────────────────────────────────────
# 2.  QAOA Circuit (Qiskit 1.x — QuantumCircuit API)
# ──────────────────────────────────────────────────────────────────────────────

def _build_qaoa_circuit(Q: np.ndarray, p: int = 1) -> QuantumCircuit:
    """
    Build QAOA circuit for QUBO problem of size n = Q.shape[0].

    Layers:
      1. Hadamard on all qubits (uniform superposition)
      2. p × (Problem unitary U_C(γ) + Mixer unitary U_B(β))

    Problem unitary encodes QUBO via:
      - RZZ(2γ * Q_ij) for i < j  (off-diagonal coupling)
      - RZ(2γ * Q_ii) for each i  (diagonal / local field)

    Mixer unitary: RX(2β) on every qubit (standard X-mixer).

    Parameters are symbolic (Qiskit Parameter objects) so we can
    bind optimal values after classical optimisation.
    """
    n = Q.shape[0]
    gammas = [Parameter(f"γ_{k}") for k in range(p)]
    betas  = [Parameter(f"β_{k}") for k in range(p)]

    qc = QuantumCircuit(n)

    # Initial state: |+⟩^n
    qc.h(range(n))

    for k in range(p):
        gamma = gammas[k]
        beta  = betas[k]

        # --- Problem unitary U_C(γ) ---
        # Diagonal terms: RZ(2γ * Q_ii)
        for i in range(n):
            if abs(Q[i, i]) > 1e-10:
                qc.rz(2 * gamma * Q[i, i], i)

        # Off-diagonal terms: CNOT + RZ + CNOT  (= RZZ)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * Q[i, j], j)
                    qc.cx(i, j)

        # --- Mixer unitary U_B(β) ---
        qc.rx(2 * beta, range(n))

    qc.measure_all()
    return qc, gammas, betas


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Classical optimiser over QAOA parameters
# ──────────────────────────────────────────────────────────────────────────────

def _cost_from_bitstring(bitstring: str, Q: np.ndarray) -> float:
    """Evaluate QUBO cost for a given bitstring (LSB = qubit 0)."""
    n = Q.shape[0]
    x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
    return float(x @ Q @ x)


def _evaluate_params(
    params: np.ndarray,
    qc_template: QuantumCircuit,
    param_list: list,
    Q: np.ndarray,
    backend: AerSimulator,
    shots: int = 1024,
) -> float:
    """Bind parameters, run circuit, compute expected cost over samples."""
    param_dict = {p: v for p, v in zip(param_list, params)}
    bound_qc = qc_template.assign_parameters(param_dict)
    transpiled = transpile(bound_qc, backend=backend, optimization_level=0)
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    total_cost = 0.0
    for bitstring, count in counts.items():
        total_cost += count * _cost_from_bitstring(bitstring, Q)
    return total_cost / shots


def _optimise_qaoa(
    qc: QuantumCircuit,
    gammas: list,
    betas: list,
    Q: np.ndarray,
    backend: AerSimulator,
    shots: int = 1024,
    maxiter: int = 60,
    n_restarts: int = 3,
) -> tuple[np.ndarray, float]:
    """
    Minimise QAOA expected cost using scipy COBYLA with multiple restarts.
    Runs COBYLA from `n_restarts` independent random starting points and
    returns the parameters that achieved the lowest expected cost.
    """
    from scipy.optimize import minimize

    p = len(gammas)
    param_list = gammas + betas
    best_params = None
    best_fun = float("inf")

    for restart in range(n_restarts):
        x0 = np.random.uniform(0, np.pi, size=2 * p)
        result = minimize(
            _evaluate_params,
            x0,
            args=(qc, param_list, Q, backend, shots),
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.5},
        )
        if result.fun < best_fun:
            best_fun = result.fun
            best_params = result.x
        print(f"[qaoa_solver]   restart {restart+1}/{n_restarts}: fun={result.fun:.4f}")

    return best_params, best_fun


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Bitstring → VRP Routes decoder
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_neighbour_route(group: list, depot: int, G) -> list:
    """
    Build a route through `group` using nearest-neighbour heuristic,
    starting and ending at depot.
    This is much better than sorting by depot distance.
    """
    if not group:
        return [depot, depot]
    unvisited = list(group)
    route = [depot]
    current = depot
    while unvisited:
        nearest = min(unvisited, key=lambda c: G[current][c].get("weight", float("inf")))
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    route.append(depot)
    return route


def _two_opt(route: list, G) -> list:
    """
    Apply 2-opt local search to improve a single vehicle route.
    Repeatedly reverses sub-segments until no improvement is found.
    Runs in O(n^2) per pass — fast enough for small routes.
    """
    if len(route) <= 3:   # depot + 0 or 1 customer + depot — nothing to swap
        return route

    def route_cost(r):
        return sum(
            G[r[i]][r[i+1]].get("weight", 0)
            for i in range(len(r) - 1)
            if r[i] != r[i+1]
        )

    improved = True
    best = list(route)
    while improved:
        improved = False
        for i in range(1, len(best) - 2):          # skip depot at 0
            for j in range(i + 1, len(best) - 1):  # skip depot at -1
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_cost(candidate) < route_cost(best) - 1e-9:
                    best = candidate
                    improved = True
    return best


def _bitstring_to_routes(
    bitstring: str,
    customers: list,
    instance: VRPInstance,
) -> dict:
    """
    Decode a QUBO bitstring as a 2-vehicle VRP assignment.

    x_i = 1  →  customer i assigned to Vehicle-1
    x_i = 0  →  customer i assigned to Vehicle-2

    Improvements vs. original:
      - Routes are built with nearest-neighbour ordering (not depot-distance sort)
      - 2-opt local search is applied to each route after construction
    """
    G = instance.graph
    depot = instance.depot
    n = len(customers)
    x = [int(b) for b in reversed(bitstring)]

    group1 = [customers[i] for i in range(n) if x[i] == 1]
    group2 = [customers[i] for i in range(n) if x[i] == 0]

    # Step 1: nearest-neighbour route construction
    route1 = _nearest_neighbour_route(group1, depot, G)
    route2 = _nearest_neighbour_route(group2, depot, G)

    # Step 2: 2-opt improvement
    route1 = _two_opt(route1, G)
    route2 = _two_opt(route2, G)

    total_cost = 0.0
    for route in (route1, route2):
        for i in range(len(route) - 1):
            if route[i] != route[i + 1]:           # skip depot→depot self-loop
                total_cost += G[route[i]][route[i + 1]].get("weight", 0)

    return {
        "routes": {"vehicle_1": route1, "vehicle_2": route2},
        "total_cost": round(total_cost, 4),
    }


def _best_assignment(
    counts: dict,
    customers: list[int],
    instance: VRPInstance,
) -> tuple[str, float]:
    """Return the bitstring with the lowest actual VRP route cost (penalizing capacity violations)."""
    best_bs = None
    # We track penalized cost to find the best valid route, but return actual
    best_penalized_cost = float("inf")
    final_actual_cost = float("inf")
    
    for bitstring in counts:
        x = [int(b) for b in reversed(bitstring)]
        
        # 1. Check if the quantum state violated capacity limits
        load1 = sum(instance.demands[customers[i]] for i in range(len(customers)) if x[i] == 1)
        load2 = sum(instance.demands[customers[i]] for i in range(len(customers)) if x[i] == 0)
        
        penalty = 0.0
        if load1 > instance.capacity:
            penalty += (load1 - instance.capacity) * 1000
        if load2 > instance.capacity:
            penalty += (load2 - instance.capacity) * 1000

        # 2. Decode the actual Euclidean travel distance
        decoded = _bitstring_to_routes(bitstring, customers, instance)
        actual_cost = decoded["total_cost"]
        
        # 3. Selection is based on travel cost + capacity penalty
        penalized_cost = actual_cost + penalty
        
        if penalized_cost < best_penalized_cost:
            best_penalized_cost = penalized_cost
            best_bs = bitstring
            # We return the purely penalized cost so the UI reflects the infeasibility 
            # if no valid route was ever found by the quantum circuit
            final_actual_cost = penalized_cost if penalty > 0 else actual_cost
            
    return best_bs, final_actual_cost


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Noise Model
# ──────────────────────────────────────────────────────────────────────────────

def _build_noise_model(error_rate: float = 0.01) -> NoiseModel:
    """
    Simple depolarising noise model (mimics real hardware noise).

    - Single-qubit gate error: error_rate
    - Two-qubit gate error:    2 * error_rate
    """
    noise_model = NoiseModel()
    single_error = depolarizing_error(error_rate, 1)
    two_error     = depolarizing_error(2 * error_rate, 2)

    noise_model.add_all_qubit_quantum_error(single_error, ["u1", "u2", "u3", "rz", "rx", "ry"])
    noise_model.add_all_qubit_quantum_error(two_error,    ["cx"])
    return noise_model


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Public API
# ──────────────────────────────────────────────────────────────────────────────

def solve_vrp_qaoa(
    vrp_instance: VRPInstance,
    p: int = 1,
    backend: str = "aer_simulator",
    noisy: bool = False,
    error_rate: float = 0.01,
    shots: int = 1024,
    maxiter: int = 60,
    seed: int = 42,
) -> dict:
    """
    Solve a VRP instance using QAOA.

    Parameters
    ----------
    vrp_instance : VRPInstance
        The VRP problem (2-vehicle; for more vehicles use a multi-partition QUBO).
    p            : int
        QAOA depth (number of problem+mixer layer pairs). Default 1.
    backend      : str
        Qiskit backend name. Currently only 'aer_simulator' is supported locally.
    noisy        : bool
        If True, apply a depolarising noise model (simulates hardware noise).
    error_rate   : float
        Depolarising error rate per gate (used only when noisy=True). Default 0.01.
    shots        : int
        Measurement shots per circuit evaluation. Default 1024.
    maxiter      : int
        COBYLA optimiser max iterations. Default 60.
    seed         : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        routes      : {'vehicle_1': List[int], 'vehicle_2': List[int]}
        best_cost   : float   (total route travel cost)
        runtime_s   : float   (wall-clock seconds)
        backend     : str
        method      : 'qaoa_ideal' | 'qaoa_noisy'
        bitstring   : str     (best measurement bitstring)
        p           : int
    """
    np.random.seed(seed)
    t0 = time.perf_counter()

    # --- Build backend ---
    if noisy:
        noise_model = _build_noise_model(error_rate)
        sim = AerSimulator(noise_model=noise_model)
        method_name = "qaoa_noisy"
    else:
        sim = AerSimulator()
        method_name = "qaoa_ideal"

    # --- Build QUBO ---
    Q, customers = _build_qubo(vrp_instance)
    n = len(customers)
    print(f"[qaoa_solver] QUBO size: {n}×{n}  |  customers: {customers}")

    # --- Build QAOA Circuit ---
    qc, gammas, betas = _build_qaoa_circuit(Q, p=p)
    print(f"[qaoa_solver] Circuit depth={qc.depth()}  qubits={qc.num_qubits}")

    # --- Classical Optimisation of γ, β (with multiple restarts) ---
    print(f"[qaoa_solver] Optimising QAOA params (COBYLA, maxiter={maxiter}, 3 restarts) ...")
    opt_params, _ = _optimise_qaoa(qc, gammas, betas, Q, sim, shots=shots,
                                    maxiter=maxiter, n_restarts=3)

    # --- Final sampling with optimal params ---
    param_dict = {p_sym: v for p_sym, v in zip(gammas + betas, opt_params)}
    bound_qc = qc.assign_parameters(param_dict)
    transpiled = transpile(bound_qc, backend=sim, optimization_level=0)
    job = sim.run(transpiled, shots=shots * 4)   # more shots for final answer
    counts = job.result().get_counts()

    # --- Decode best bitstring ---
    best_bs, best_cost = _best_assignment(counts, customers, vrp_instance)
    decoded = _bitstring_to_routes(best_bs, customers, vrp_instance)

    runtime = time.perf_counter() - t0
    print(f"[qaoa_solver] Done. best_cost={best_cost:.4f}  runtime={runtime:.2f}s")

    return {
        "routes": decoded["routes"],
        "best_cost": best_cost,
        "runtime_s": round(runtime, 4),
        "backend": backend,
        "method": method_name,
        "bitstring": best_bs,
        "p": p,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  QuantumRoute-AI — QAOA VRP Solver Demo")
    print("=" * 60)

    instance = build_vrp_instance(n_customers=4, n_vehicles=2, capacity=10, seed=42)
    print(f"\nVRP Instance: {len(instance.graph.nodes)} nodes, depot=0, "
          f"vehicles={instance.n_vehicles}, capacity={instance.capacity}")
    print(f"Demands: {instance.demands}")

    # Ideal simulation
    print("\n--- QAOA (Ideal Simulator) ---")
    ideal = solve_vrp_qaoa(instance, p=1, backend="aer_simulator",
                            noisy=False, maxiter=60, shots=1024)
    print(f"  Vehicle 1: {ideal['routes']['vehicle_1']}")
    print(f"  Vehicle 2: {ideal['routes']['vehicle_2']}")
    print(f"  Best cost: {ideal['best_cost']}")
    print(f"  Runtime  : {ideal['runtime_s']}s")
    print(f"  Method   : {ideal['method']}")

    # Noisy simulation
    print("\n--- QAOA (Noisy Simulator, error_rate=0.01) ---")
    noisy = solve_vrp_qaoa(instance, p=1, backend="aer_simulator",
                            noisy=True, error_rate=0.01, maxiter=60, shots=1024)
    print(f"  Vehicle 1: {noisy['routes']['vehicle_1']}")
    print(f"  Vehicle 2: {noisy['routes']['vehicle_2']}")
    print(f"  Best cost: {noisy['best_cost']}")
    print(f"  Runtime  : {noisy['runtime_s']}s")
    print(f"  Method   : {noisy['method']}")
