from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_tsp_graph(n_cities: int, seed: int = 42) -> nx.Graph:
    np.random.seed(seed)
    G = nx.complete_graph(n_cities)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.randint(1, 20)
    return G


def draw_tsp_graph(G: nx.Graph, filename: str) -> None:
    pos = nx.spring_layout(G, seed=42)
    weights = nx.get_edge_attributes(G, "weight")
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        font_size=12,
        width=2,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def solve_tsp_qaoa(G: nx.Graph, reps: int = 1) -> dict:
    tsp = Tsp(G)
    qp = tsp.to_quadratic_program()
    print("TSP QuadraticProgram (num vars):", qp.get_num_vars())

    # Convert to QUBO so QAOA can run
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)
    print("QUBO (num vars):", qubo.get_num_vars())

    sampler = Sampler()
    optimizer = COBYLA(maxiter=200)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    solver = MinimumEigenOptimizer(qaoa)

    result = solver.solve(qubo)
    route = tsp.interpret(result)
    cost = result.fval

    print("Best TSP route:", route)
    print("Total cost:", cost)

    return {"route": route, "cost": cost, "result": result}


def main() -> None:
    G = build_tsp_graph(4)  # 4 cities
    draw_tsp_graph(G, "tsp_4_cities_graph.png")

    solution = solve_tsp_qaoa(G, reps=1)

    print("\nGenerated files:")
    print("- outputs/tsp_4_cities_graph.png")


if __name__ == "__main__":
    import numpy as np
    main()
