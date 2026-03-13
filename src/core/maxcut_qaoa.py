from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut
from qiskit_aer import AerSimulator
from qiskit_algorithms import NumPyMinimumEigensolver


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 0, 1.0),
        (0, 2, 1.0),
    ])
    return graph


def draw_graph(graph: nx.Graph, filename: str) -> None:
    pos = nx.spring_layout(graph, seed=42)
    weights = nx.get_edge_attributes(graph, "weight")

    plt.figure(figsize=(6, 5))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue",
            node_size=900, font_size=12, width=2)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
    plt.title("Input Graph for Max-Cut")
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def solve_classical_maxcut(graph: nx.Graph) -> dict:
    """Solve Max-Cut using classical NumPy eigensolver."""
    maxcut = Maxcut(graph)
    qp = maxcut.to_quadratic_program()

    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = solver.solve(qp)

    x = result.x
    objective = result.fval

    left_partition = [i for i, bit in enumerate(x) if bit == 0]
    right_partition = [i for i, bit in enumerate(x) if bit == 1]

    print("Classical Max-Cut Result:")
    print(f"Best bitstring: {x}")
    print(f"Objective value: {objective}")
    print(f"Partition A: {left_partition}")
    print(f"Partition B: {right_partition}")

    return {"bitstring": x, "objective": objective}


def visualize_partition(graph: nx.Graph, bitstring: list, filename: str) -> None:
    pos = nx.spring_layout(graph, seed=42)
    colors = ["tomato" if bit == 0 else "mediumseagreen" for bit in bitstring]
    weights = nx.get_edge_attributes(graph, "weight")

    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color=colors,
            node_size=1000, font_size=14, width=3)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
    plt.title("Classical Max-Cut Solution (Red=Partition A, Green=Partition B)")
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=== QuantumRoute-AI: Max-Cut (Classical Solver) ===\n")
    
    graph = build_graph()
    draw_graph(graph, "maxcut_input_graph.png")
    
    result = solve_classical_maxcut(graph)
    visualize_partition(graph, result["bitstring"], "maxcut_classical_result.png")
    
    print("\nFiles saved:")
    print("- outputs/maxcut_input_graph.png")
    print("- outputs/maxcut_classical_result.png")
    print("=== Max-Cut Classical Complete ===")


if __name__ == "__main__":
    main()
