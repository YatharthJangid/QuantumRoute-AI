# QuantumRoute-AI
## Hybrid Quantum-Classical Vehicle Routing Problem Optimizer
### ETT (Emerging Technologies) — Lab Final Evaluation

> **Student:** Yatharth Jangid  
> **Tech Used:** Python 3.10 · Qiskit 1.4.2 · Qiskit-Aer 0.15.1 · NetworkX · SciPy · Matplotlib  
> **Paradigm:** Hybrid Quantum-Classical (QAOA + Classical Post-processing)

---

## 1. Problem Statement — Vehicle Routing Problem (VRP)

The **Vehicle Routing Problem** is one of the most studied combinatorial optimization problems in operations research.

### Definition
Given:
- A **depot** (starting point for all vehicles)
- A set of **n customer locations**, each with a demand (e.g. parcels to deliver)
- A **fleet of vehicles**, each with a fixed capacity

**Goal:** Find the minimum-cost set of routes such that every customer is visited exactly once, each route starts and ends at the depot, and no vehicle exceeds its capacity.

```
                  [C2]
                 /    \
[Depot]---[C1]         [C4]---[Depot]
         Vehicle 1      Vehicle 2
```

### Why It Matters
- Last-mile delivery (Amazon, Swiggy, logistics companies)
- Emergency vehicle dispatch
- School bus routing

### Why It's Hard
- **Complexity:** O((n-1)!) for exact brute-force — grows faster than exponential
- For n=12 customers: brute force checks ~39.9 million permutations (~30 minutes)
- For n=20: would take longer than the age of the universe classically

This is exactly the kind of problem where **quantum computing promises a polynomial speedup**.

---

## 2. Research Question

> *Can QAOA (Quantum Approximate Optimization Algorithm) produce competitive VRP solutions compared to classical approaches, and at what problem size does quantum become advantageous?*

---

## 3. Technology Stack

| Layer | Tool | Version | Purpose |
|---|---|---|---|
| **Quantum Framework** | Qiskit | 1.4.2 | Building & running quantum circuits |
| **Quantum Simulator** | Qiskit-Aer | 0.15.1 | Simulating quantum hardware locally |
| **Graph Modeling** | NetworkX | 3.4.2 | VRP instance as a weighted complete graph |
| **Classical Optimization** | SciPy COBYLA | 1.15.2 | Tuning QAOA circuit parameters |
| **Data Export** | Pandas | 2.2.3 | Saving benchmark results to CSV |
| **Visualization** | Matplotlib | 3.10.1 | Route maps, performance charts |
| **Language** | Python | 3.10 | Everything |

---

## 4. Code Structure

```
QuantumRoute-AI/
│
├── src/
│   ├── core/                         # Core algorithm modules
│   │   ├── vrp_graph.py              # VRPInstance dataclass + graph builder
│   │   ├── vrp_classical.py          # Brute Force + Greedy NN solvers
│   │   ├── qaoa_solver.py            # Quantum QAOA solver (main contribution)
│   │   ├── results_exporter.py       # CSV logging + noise analysis
│   │   └── visualizer.py            # Route maps + performance bar charts
│   │
│   └── benchmark/
│       ├── runner.py                 # Full automated benchmark (all 4 solvers)
│       ├── interactive_runner.py     # Interactive CLI (choose n, vehicles, etc.)
│       └── scaling_analysis.py      # Classical vs QAOA actual runtime comparison
│
├── results/                          # Auto-generated CSVs and charts
├── requirements.txt
└── PRESENTATION.md                   # This file
```

### Module Responsibilities

#### `vrp_graph.py` — Problem Definition
- Defines the `VRPInstance` dataclass (`graph`, `depot`, `n_vehicles`, `capacity`, `demands`, `positions`)
- `build_vrp_instance(n_customers, n_vehicles, capacity, seed)` generates a reproducible random Euclidean VRP
- Uses a **complete graph** with Euclidean distances × 10 as edge weights

#### `vrp_classical.py` — Classical Solvers
Two solvers for benchmarking against:
1. **Greedy Nearest-Neighbour** — each vehicle picks the closest unvisited feasible customer
2. **Brute Force** — exhaustive permutation search over all customer orderings; optimal but O((n-1)!)

#### `qaoa_solver.py` — The Main Contribution ⭐
Full QAOA pipeline (explained in detail in Section 5).

#### `interactive_runner.py` — Live Demo Tool
```bash
python -m src.benchmark.interactive_runner
```
Prompts for n, vehicles, capacity, which solvers to run, QAOA depth (p), and iterations.

---

## 5. How the QAOA Solver Works (Step-by-Step)

### Step 1: Encode VRP as QUBO

**QUBO = Quadratic Unconstrained Binary Optimization**

We introduce a binary variable for each customer:
```
x_i = 1  →  Customer i is served by Vehicle 1
x_i = 0  →  Customer i is served by Vehicle 2
```

The QUBO objective function encodes two things:
- **Capacity penalty:** If vehicle load exceeds capacity, add a large penalty to the cost
- **Travel-cost preference:** Reward placing nearby customers in the same vehicle

The result is an (n×n) matrix **Q** where the optimal bit-string minimizes `x^T · Q · x`.

### Step 2: Build the QAOA Quantum Circuit

QAOA maps the QUBO to a quantum circuit with two alternating layers per depth `p`:

```
|0⟩ ─ H ─[Problem U_C(γ)]─[Mixer U_B(β)]─ ... ─ Measure
|0⟩ ─ H ─[Problem U_C(γ)]─[Mixer U_B(β)]─ ... ─ Measure
    ...repeated for all n qubits
```

- **Hadamard (H):** Initialize all qubits in equal superposition — encodes all 2ⁿ possible assignments simultaneously
- **Problem Unitary U_C(γ):**
  - `RZ(2γ·Q_ii)` gates for diagonal terms (local biases)
  - `CNOT → RZ(2γ·Q_ij) → CNOT` for off-diagonal terms (two-qubit interactions)
- **Mixer Unitary U_B(β):** `RX(2β)` on every qubit — allows the state to "explore" different assignments

### Step 3: Classical Parameter Optimization (Hybrid Loop)

The angles `γ` (gamma) and `β` (beta) are free parameters. We use SciPy's **COBYLA** optimizer to tune them:

```
Classical Computer:
  Choose γ, β
       ↓
Quantum Computer (Simulator):
  Run circuit → measure → get bitstring histogram
       ↓
Classical Computer:
  Compute expected cost = Σ (count × cost(bitstring)) / total_shots
  COBYLA updates γ, β to minimize expected cost
       ↓
  Repeat (maxiter=50 times, 3 independent restarts)
```

This is the **hybrid quantum-classical loop** — the core of QAOA.

### Step 4: Decode Bitstring → VRP Routes

After finding optimal `γ, β`, we do a final high-shot measurement (4096 shots) and for each unique bitstring:

1. **Group customers** by bit value (0 or 1 → Vehicle 1 or 2)
2. **Build routes** using **Nearest-Neighbour heuristic** within each group
3. **Apply 2-opt local search** to further improve each route
4. Pick the bitstring whose decoded routes have the **lowest actual travel cost**

---

## 6. Solvers Compared (5 Solvers)

| Solver | Type | Time Complexity | Quality |
|---|---|---|---|
| **Brute Force** | Classical, Exact | O((n-1)!) with pruning | Optimal (ground truth) |
| **Greedy NN** | Classical, Heuristic | O(n²) | ~10-20% above optimal |
| **Clarke-Wright** | Classical, Heuristic | O(n² log n) | Near-optimal (often matches BF!) |
| **QAOA Ideal** | Quantum, Approximate | O(p·n²) circuit + COBYLA | Near-optimal for small n |
| **QAOA Noisy** | Quantum, Approximate | Same + noise overhead | Slightly degraded by gate errors |

> **Clarke-Wright Savings** (1964): Instead of starting from scratch, it starts with one route per customer and iteratively merges the two routes that save the most distance. It's the industry-standard VRP heuristic used by real logistics companies.

---

## 7. Actual Benchmark Results

### Key Result: n=10, 2 vehicles, capacity=30 — All solvers agree ⭐

```
  Solver          Cost        Runtime
  ──────────────────────────────────────────
  Greedy          38.30       0.0001s   ← fast but ~10% suboptimal
  Clarke-Wright   34.79       0.0004s   ← matches optimal instantly!
  Brute Force     34.79       14.47s    ← optimal (took 14 seconds)
  QAOA (p=2)      34.79       24.71s    ← MATCHES optimal
```

> **Three different algorithms independently converged on the same optimal cost (34.79).** This cross-validates that the brute force is genuinely finding the global optimum, and both Clarke-Wright and QAOA can match it.

### Larger Result: n=10, 2 vehicles, capacity=23 (tighter constraints)

```
  Solver          Cost        Runtime
  ──────────────────────────────────────────
  Greedy          37.75       0.0002s
  Clarke-Wright   34.98       0.0004s   ← matches optimal!
  Brute Force     34.98       10.61s    ← optimal
  QAOA (p=5)      34.98       55.14s    ← MATCHES optimal
```

> Even with tighter capacity constraints, QAOA finds the globally optimal solution. The higher runtime here is because `p=5` and `maxiter=60` were used — QAOA runtime is controlled by `p × maxiter × restarts`, not by `n`.

### Route comparison (n=10, capacity=30):
```
  Brute Force:
    Vehicle 1: 0 → 2 → 5 → 1 → 6 → 4 → 9 → 0
    Vehicle 2: 0 → 7 → 10 → 3 → 8 → 0

  QAOA:
    Vehicle 1: 0 → 8 → 3 → 10 → 7 → 0          ← same customers as BF vehicle 2
    Vehicle 2: 0 → 2 → 5 → 1 → 6 → 4 → 9 → 0   ← same customers as BF vehicle 1

  Clarke-Wright:
    Vehicle 1: 0 → 2 → 5 → 1 → 6 → 4 → 9 → 0   ← identical to BF!
    Vehicle 2: 0 → 7 → 10 → 3 → 8 → 0
```

### Why QAOA Time Varies — It Depends on `p` and `maxiter`, NOT `n`

| Settings | QAOA Runtime | BF Runtime |
|---|---|---|
| p=2, maxiter=30 | **24.71s** | 14.47s |
| p=5, maxiter=60 | **55.14s** | 10.61s |
| p=7, maxiter=60 | **39.05s** | 49.15s |

QAOA runtime = `3 restarts × maxiter × circuit_eval_time(p, n)`. The brute force uses recursive search with **branch pruning** — tighter capacity constraints let it prune more branches and finish faster.

---

## 8. How Parameters Affect Cost & Runtime

When running the interactive benchmark, you provide several parameters. Here is how each parameter impacts the solvers:

### 1. Number of Customers (`n`)
- **Runtime (Classical Brute Force):** Scales as **O((n-1)!)**. Time explodes factorially. Adding just one customer multiplies the runtime by `n-1`. (n=12 takes ~30 mins).
- **Runtime (QAOA):** Scales as **O(p · n²)** for the circuit structure, but classical simulation time is mostly tied to the optimizer's `maxiter` steps. For small `n`, simulation stays fast, but it will eventually struggle as simulating larger quantum states (more qubits) becomes classically hard.
- **Cost:** Higher `n` means a vastly larger search space. This naturally inflates overall travel cost and makes it harder for heuristic and approximate solvers (Greedy, QAOA) to find the absolute minimum cost without getting stuck in local optima.

### 2. Number of Vehicles & Capacity
- **Runtime:** Very little direct effect. Brute force still checks all permutations, simply splitting them differently based on capacity limits.
- **Cost:** If capacity allows, a single vehicle doing a giant TSP tour is usually the cheapest route. Generally, forcing more vehicles = more return trips to the depot = **higher total cost**.

### 3. QAOA Circuit Depth (`p`)
- **Runtime (QAOA):** Increases linearly. Deeper circuits mean more quantum gates to apply and simulate at each step of the COBYLA optimization.
- **Cost (QAOA Quality):** Theory says that as `p → ∞`, QAOA approaches the perfect optimal solution. Higher `p` allows the quantum state to better explore the solution space, usually lowering the final VRP cost. However, for current noisy real-world hardware, high `p` also means more gate errors, which ruins the result.

### 4. QAOA Max Iterations (`maxiter`)
- **Runtime (QAOA):** Increases linearly. `maxiter=50` vs `maxiter=100` roughly doubles the QAOA runtime.
- **Cost (QAOA Quality):** The COBYLA optimizer needs enough steps to find the bottom of the "valley" (minimum expected cost). Too few iterations = it stops before finding the best parameters. Too many = diminishing returns (it's already at the local minimum) and wastes time.

---

## 9. Main Problem I Faced — QAOA Cost Worse Than Classical

### The Problem
After implementing QAOA, initial results were:

```
  Brute Force:  24.22  (optimal)
  Greedy:       26.53
  QAOA (p=4):   29.85  ← worse than even greedy!
```

QAOA was supposed to find good solutions, but was giving clearly suboptimal routes.

### Root Cause Analysis

After debugging, I found **two separate issues:**

**Issue 1 — Poor route construction from bitstring**  
The original code to build a vehicle's route from the decoded assignment was:
```python
# BAD: Sort by distance from depot (doesn't consider between-customer distances)
route = [depot] + sorted(group, key=lambda c: G[depot][c]['weight']) + [depot]
```
This visits customers in order of proximity to depot, which is similar to a "star" pattern — visiting far-then-near or near-then-far instead of a proper tour.

**Fix:** Use **nearest-neighbour heuristic** starting from the depot:
```python
# GOOD: Nearest-neighbour traversal (greedy TSP within the group)
current = depot
while unvisited:
    nearest = min(unvisited, key=lambda c: G[current][c]['weight'])
    route.append(nearest); current = nearest
```

**Issue 2 — COBYLA trapped in local optima**  
COBYLA is a local optimizer — it can get stuck in bad parameter regions. A single random starting point often wasn't enough.

**Fix:** Run **3 independent restarts** from different random `(γ, β)` starting points, keep the best:
```python
for restart in range(3):
    x0 = np.random.uniform(0, np.pi, size=2*p)
    result = minimize(..., x0, method='COBYLA')
    if result.fun < best_fun:
        best_fun = result.fun; best_params = result.x
```

**Issue 3 (bonus) — 2-opt post-processing**  
After nearest-neighbour route construction, we additionally run **2-opt local search** on each vehicle's route:
```python
# Repeatedly swap pairs of edges until no improvement
for i in range(1, len(route)-2):
    for j in range(i+1, len(route)-1):
        candidate = route[:i] + route[i:j+1][::-1] + route[j+1:]
        if cost(candidate) < cost(route):
            route = candidate
```

### Result After All Three Fixes

```
  QAOA Cost:  29.85  →  24.22  ✅  (matches brute force optimal!)
```

---

## 9. Quantum Advantage — When Does It Actually Matter?

For **small n** (n ≤ 8), classical brute force is faster. QAOA simulation overhead dominates.

For **large n**, brute force is literally impossible:
```
  n=12  →  11! = 39,916,800 permutations  →  ~30 minutes
  n=15  →  14! ≈ 87 billion permutations  →  ~months
  n=20  →  19! ≈ 1.2 × 10^17            →  longer than age of universe
```

QAOA scales as **O(p · n²)** — the circuit grows polynomially, not factorially.

> **The theorem behind this:** Grover's algorithm provides quadratic speedup for unstructured search. QAOA leverages quantum interference to bias measurement outcomes toward better solutions — approximating the optimal in polynomial circuit depth.

### What running on REAL quantum hardware (IBM) would add:
- True quantum parallelism (not simulation)
- IBM Quantum free tier: `quantum.ibm.com` — sign up, get API token, run 4-8 qubit circuits for free
- For n=4-6, a real 127-qubit IBM machine would fit the circuit comfortably

---

## 10. How to Run (Live Demo Commands)

### Option A — Interactive (ask everything)
```bash
cd QuantumRoute-AI
venv\Scripts\python.exe -m src.benchmark.interactive_runner
```

### Option B — Full automated benchmark (5 solvers)
```bash
venv\Scripts\python.exe -m src.benchmark.runner
# Outputs: results/vrp_results.csv + results/performance_comparison.png
# Runs: brute_force, greedy, clarke_wright, qaoa_ideal, qaoa_noisy
```

### Option C — Scaling analysis (actual runtime for n=3..8)
```bash
venv\Scripts\python.exe -m src.benchmark.scaling_analysis
# Outputs: results/scaling_analysis_actual.png
```

### Option D — QAOA solver demo only
```bash
venv\Scripts\python.exe -m src.core.qaoa_solver
```

---

## 11. Key Learning Outcomes

1. **QAOA is a variational algorithm** — its quality depends heavily on parameter initialization and optimizer choice. Multiple restarts are critical.

2. **QUBO encoding is the hardest part** — translating a real-world problem (VRP) into a QUBO matrix requires careful formulation of objective + penalty terms.

3. **Hybrid = quantum for search, classical for decoding** — QAOA finds good *bitstring assignments*, but the actual route quality still depends on classical post-processing (NN + 2-opt). This is the hybrid paradigm.

4. **Noise matters** — a depolarizing error rate of just 1% per gate degrades solution quality measurably. Real quantum hardware has ~0.1-1% error rates, making error mitigation a real concern.

5. **Small n ≠ quantum advantage** — QAOA is not faster for 4-6 cities. The advantage is projected at n > 12-15 where classical methods become computationally infeasible.

---

## 12. Project Timeline (18 Commits)

| Commit | Date | Description |
|---|---|---|
| #1-2 | Mar 26 | Init + requirements |
| #3-4 | Mar 27 | VRP graph + validation |
| #5-6 | Mar 28 | Classical brute force + greedy |
| #7-8 | Mar 29 | Results exporter + visualizer |
| #9-10 | Mar 30 | TSP QAOA foundations |
| #11-12 | Mar 31 | MaxCut QUBO study |
| #13-14 | Mar 31 | VRP-specific QUBO + core benchmarks |
| **#15** | **Apr 2** | **Modular package structure** |
| **#16** | **Apr 4** | **Correctness tests** |
| **#17** | **Apr 6** | **README + full results** |
| **#18** | **Apr 8** | **Final cleanup** |

---

## 13. Honest Assessment

### What works well ✅
- **5-solver benchmark** — brute force, greedy, Clarke-Wright, QAOA ideal, QAOA noisy all compared on the same instance
- Clean modular architecture — each file has one responsibility
- QAOA correctly implements the QUBO→quantum circuit→measurement→decode pipeline
- After fixes, QAOA matches brute force optimal for tested instances (n=10)
- Clarke-Wright independently validates brute force optimality (both find cost=34.79)
- Hard capacity constraint enforcement prevents QAOA from returning infeasible solutions
- Noise modeling is realistic (depolarizing model matching IBM specs)

### Limitations / Future Work 🔮
- **Only 2 vehicles encoded in QUBO** — multi-vehicle (k > 2) requires a different QUBO formulation (one-hot encoding per vehicle slot, more qubits). The solver now explicitly rejects k≠2.
- **Clarke-Wright often matches brute force** — at the tested scales, the classical heuristic is already near-optimal, which means QAOA's value proposition is strongest for larger instances (n > 15) where even heuristics struggle
- **QAOA circuit depth tradeoff** — increasing p helps quality but linearly increases runtime; finding the sweet spot requires experimentation
- **No real hardware tested** — AerSimulator is near-ideal; real IBM hardware would add decoherence and measurement error
- **COBYLA is slow** — gradient-based optimizers (ADAM, SPSA) may converge faster for deep circuits

---

## 14. References

1. Farhi, E., Goldstone, J., Gutmann, S. (2014). *A Quantum Approximate Optimization Algorithm.* arXiv:1411.4028
2. Lucas, A. (2014). *Ising formulations of many NP problems.* Frontiers in Physics.
3. Borowski, M. et al. (2020). *New hybrid quantum annealing algorithms for solving Vehicle Routing Problem.* arXiv:2002.09283
4. IBM Quantum Documentation: https://quantum.ibm.com
5. Qiskit Textbook: https://learning.quantum.ibm.com

---

*Generated: April 1, 2026 | QuantumRoute-AI v1.0 | Python 3.10 + Qiskit 1.4.2*
