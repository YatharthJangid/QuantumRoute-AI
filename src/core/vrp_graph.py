from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

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


def analyze_vrp_instance(instance: VRPInstance) -> dict:
    """
    Inspect a VRP instance and return structural and feasibility metadata.

    `is_valid` means the instance is well-formed.
    `is_feasible` additionally means the declared fleet can serve all customers.
    """
    issues = []
    infeasible_reasons = []

    if instance.graph is None:
        issues.append("graph is missing")
        return {
            "customers": [],
            "total_demand": 0,
            "max_capacity": 0,
            "customers_over_capacity": [],
            "issues": issues,
            "infeasible_reasons": infeasible_reasons,
            "is_valid": False,
            "is_feasible": False,
        }

    if instance.depot not in instance.graph:
        issues.append(f"depot {instance.depot} is not present in the graph")

    if instance.n_vehicles < 0:
        issues.append("number of vehicles cannot be negative")

    if instance.capacity < 0:
        issues.append("vehicle capacity cannot be negative")

    customers = [n for n in sorted(instance.graph.nodes()) if n != instance.depot]

    missing_positions = [n for n in instance.graph.nodes if n not in instance.positions]
    if missing_positions:
        issues.append(f"positions missing for nodes: {missing_positions}")

    missing_demands = [n for n in instance.graph.nodes if n not in instance.demands]
    if missing_demands:
        issues.append(f"demands missing for nodes: {missing_demands}")

    negative_demands = [
        n for n, demand in instance.demands.items()
        if n in instance.graph and demand < 0
    ]
    if negative_demands:
        issues.append(f"negative demands found at nodes: {negative_demands}")

    if instance.depot in instance.demands and instance.demands[instance.depot] != 0:
        issues.append("depot demand must be 0")

    total_demand = sum(instance.demands.get(node, 0) for node in customers)
    max_capacity = instance.n_vehicles * instance.capacity

    customers_over_capacity = [
        node for node in customers
        if instance.demands.get(node, 0) > instance.capacity
    ]
    if customers_over_capacity:
        infeasible_reasons.append(
            f"customer demands exceed vehicle capacity: {customers_over_capacity}"
        )

    if customers and instance.n_vehicles == 0:
        infeasible_reasons.append("no vehicles available to serve customers")

    if total_demand > max_capacity:
        infeasible_reasons.append(
            f"total demand {total_demand} exceeds fleet capacity {max_capacity}"
        )

    is_valid = not issues
    is_feasible = is_valid and not infeasible_reasons

    return {
        "customers": customers,
        "total_demand": total_demand,
        "max_capacity": max_capacity,
        "customers_over_capacity": customers_over_capacity,
        "issues": issues,
        "infeasible_reasons": infeasible_reasons,
        "is_valid": is_valid,
        "is_feasible": is_feasible,
    }


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

    positions = {0: (0.5, 0.5)}
    for i in range(1, total_nodes):
        positions[i] = (np.random.uniform(0, 1), np.random.uniform(0, 1))

    G = nx.complete_graph(total_nodes)
    for u, v in G.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        dist = round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 10, 2)
        G[u][v]["weight"] = dist

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


def validate_vrp(instance: VRPInstance, raise_on_error: bool = False) -> dict:
    """Print validation checks and optionally raise when the instance is unusable."""
    analysis = analyze_vrp_instance(instance)

    print("=== VRP Instance Validation ===")
    print(f"  Nodes       : {list(instance.graph.nodes)}")
    print(f"  Depot       : {instance.depot}")
    print(f"  Vehicles    : {instance.n_vehicles}")
    print(f"  Capacity    : {instance.capacity} per vehicle")
    print(f"  Demands     : {instance.demands}")
    print(f"  Total demand: {analysis['total_demand']}")
    print(f"  Max capacity: {analysis['max_capacity']}")

    if analysis["issues"]:
        print("  INVALID:")
        for issue in analysis["issues"]:
            print(f"    - {issue}")
    elif analysis["infeasible_reasons"]:
        print("  WARNING: Instance is structurally valid but infeasible.")
        for reason in analysis["infeasible_reasons"]:
            print(f"    - {reason}")
    else:
        print("  Status: Feasible instance.")

    if raise_on_error and not analysis["is_feasible"]:
        reasons = analysis["issues"] + analysis["infeasible_reasons"]
        raise ValueError("; ".join(reasons))

    return analysis


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
        G,
        pos,
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
    plt.title("VRP Instance - Depot + Customers with Demands")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/{filename}")


if __name__ == "__main__":
    instance = build_vrp_instance(n_customers=5, n_vehicles=2, capacity=10)
    validate_vrp(instance)
    draw_vrp_instance(instance)
