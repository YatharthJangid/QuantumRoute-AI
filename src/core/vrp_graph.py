from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class VRPInstance:
    graph: nx.Graph
    depot: int
    n_vehicles: int
    capacity: int
    demands: Dict[int, int]
    positions: Dict[int, Tuple[float, float]]


def build_vrp_instance(
    n_customers: int = 5,
    n_vehicles: int = 2,
    capacity: int = 10,
    seed: int = 42,
) -> VRPInstance:
    """
    Build a VRP instance with:
      - Node 0 as the depot (demand = 0)
      - Nodes 1..n_customers as delivery locations
      - Random demands between 2 and 5 per customer
      - Euclidean distances as edge weights
    """
    np.random.seed(seed)
    total_nodes = n_customers + 1

    # Random 2D positions (depot at center)
    positions = {0: (0.5, 0.5)}
    for i in range(1, total_nodes):
        positions[i] = (np.random.uniform(0, 1), np.random.uniform(0, 1))

    # Build complete graph with Euclidean weights
    G = nx.complete_graph(total_nodes)
    for u, v in G.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        dist = round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 10, 2)
        G[u][v]["weight"] = dist

    # Assign demands
    demands = {0: 0}
    for i in range(1, total_nodes):
        demands[i] = np.random.randint(2, 6)

    return VRPInstance(
        graph=G,
        depot=0,
        n_vehicles=n_vehicles,
        capacity=capacity,
        demands=demands,
        positions=positions,
    )


def validate_vrp(instance: VRPInstance) -> None:
    """Basic feasibility checks on the VRP instance."""
    total_demand = sum(
        d for node, d in instance.demands.items() if node != instance.depot
    )
    max_capacity = instance.n_vehicles * instance.capacity

    print("=== VRP Instance Validation ===")
    print(f"  Nodes       : {list(instance.graph.nodes)}")
    print(f"  Depot       : {instance.depot}")
    print(f"  Vehicles    : {instance.n_vehicles}")
    print(f"  Capacity    : {instance.capacity} per vehicle")
    print(f"  Demands     : {instance.demands}")
    print(f"  Total demand: {total_demand}")
    print(f"  Max capacity: {max_capacity}")

    if total_demand > max_capacity:
        print("  WARNING: Total demand exceeds fleet capacity — infeasible instance.")
    else:
        print("  Status: Feasible instance.")


def draw_vrp_instance(instance: VRPInstance, filename: str = "vrp_instance.png") -> None:
    G = instance.graph
    pos = instance.positions

    node_colors = [
        "gold" if n == instance.depot else "lightblue"
        for n in G.nodes()
    ]
    labels = {
        n: f"{n}\n(depot)" if n == instance.depot else f"{n}\nd={instance.demands[n]}"
        for n in G.nodes()
    }
    edge_labels = nx.get_edge_attributes(G, "weight")

    plt.figure(figsize=(8, 7))
    nx.draw(
        G, pos,
        labels=labels,
        node_color=node_colors,
        node_size=1200,
        font_size=9,
        width=0.5,
        edge_color="lightgray",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    depot_patch = mpatches.Patch(color="gold", label="Depot (node 0)")
    customer_patch = mpatches.Patch(color="lightblue", label="Customer (d=demand)")
    plt.legend(handles=[depot_patch, customer_patch], loc="upper right")
    plt.title("VRP Instance — Depot + Customers with Demands")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/{filename}")


if __name__ == "__main__":
    instance = build_vrp_instance(n_customers=5, n_vehicles=2, capacity=10)
    validate_vrp(instance)
    draw_vrp_instance(instance)
