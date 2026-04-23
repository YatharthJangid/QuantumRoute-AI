# QuantumRoute-AI 🚀

> **Hybrid Quantum-Classical Vehicle Routing Problem Optimizer**
> *Developed for ETT (Emerging Technologies) Lab Final Evaluation*

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.4.2-purple.svg)](https://qiskit.org/)

**QuantumRoute-AI** investigates whether QAOA (Quantum Approximate Optimization Algorithm) can produce competitive solutions for the Vehicle Routing Problem (VRP) compared to classical heuristics, and empirically analyzes the quantum advantage threshold.

---

## 📖 The Problem: Vehicle Routing

The VRP requires finding optimal delivery routes for a fleet of vehicles servicing a set of customers. Brute-forcing this geometrically explodes to an $O((n-1)!)$ complexity. For just 20 customers, exact classical solvers take longer than the age of the universe. 

This project tackles the VRP using **QAOA**, demonstrating how quantum computing can potentially offer a polynomial speedup over factorial growth boundaries.

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10 | Core logic and scripting |
| **Quantum Framework** | Qiskit | 1.4.2 | Quantum circuit composition |
| **Circuit Simulation** | Qiskit-Aer | 0.15.1 | High-performance local simulation |
| **Classical Optimizer**| SciPy (COBYLA)| 1.15.2 | Tuning hybrid loop parameters (`γ`, `β`)|
| **Graph Modeling** | NetworkX | 3.4.2 | Modeling the VRP graphs | 
| **Visualization** | Matplotlib | 3.10.1 | Outputting final routes and charts |

---

## ⚙️ How It Works

1. **QUBO Encoding**: Encodes the problem as a Quadratic Unconstrained Binary Optimization (QUBO) model. Modulates variables representing exactly 2 vehicles with strict capacity limits.
2. **QAOA Circuit Construction**: Compiles the QUBO matrices into Phase Separation and Mixing unitary gates.
3. **Hybrid Optimization**: Offloads parameter tuning (`γ`, `β`) to COBYLA wrapped around the simulated quantum executions to minimize the expected route cost.
4. **Classical Decoding**: Processes the measured bitstrings using Nearest-Neighbor generation coupled with 2-opt search space refinement.

---

## 📂 Project Structure
```text
QuantumRoute-AI/
├── src/
│   ├── core/                        # Core algorithmic modules
│   │   ├── vrp_graph.py             # Data models & underlying problem construction
│   │   ├── vrp_classical.py         # Baseline benchmarks (Brute Force, Greedy NN)
│   │   ├── qaoa_solver.py           # Quantum-based QAOA solver
│   │   ├── results_exporter.py      # I/O reporting and CSV dumps
│   │   └── visualizer.py            # Route and metric rendering
│   │
│   └── benchmark/
│       ├── runner.py                # Automated cross-solver evaluation suite
│       ├── interactive_runner.py    # Highly configurable interactive prompt
│       └── scaling_analysis.py      # Runtime crossover mapping
│
├── results/                         # Generated plots and benchmarking metadata
├── PRESENTATION.md                  # Detailed exposition & project insights
└── requirements.txt                 # Dependency freeze
```

---

## 🚀 Installation & Usage

1. **Clone & Setup Environment**
   ```bash
   git clone <your-repo-url>
   cd QuantumRoute-AI
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Interactive Benchmark**
   Engage with the comprehensive evaluation tool:
   ```bash
   python -m src.benchmark.interactive_runner
   ```

4. **Execute specific modules**
   - **Head-to-Head Tests**: `python -m src.benchmark.runner`
   - **Single QAOA Demo**: `python -m src.core.qaoa_solver`
   - **Real-World Scaling Check**: `python -m src.benchmark.scaling_analysis`

---

## 📊 Key Findings

In benchmarks of size $n=10$ with `capacity=30`:
- **Brute Force Solutions** confirmed globally optimal route sets with a cost of `34.79`.
- **QAOA** (configured at $p=2$) consistently replicated the pristine optimal solution of `34.79` over local simulation instances.
- **Clarke-Wright** and **Greedy NN** algorithms also ran as comparative classical baselines. 

For smaller numbers of customers ($n \\le 8$), brute force is faster. As graph sizes scale past $n=12$, classical solutions begin to dramatically fail off polynomial scaling curves, displaying the structural advantage of QAOA representations.

---
*Created by Yatharth Jangid for the ETT (Emerging Technologies) lab evaluation.*