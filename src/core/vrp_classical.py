import itertools
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from src.core.vrp_graph import VRPInstance, build_vrp_instance


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def greedy_vrp(instance: VRPInstance) -> Dict:
    """
    Nearest-neighbour greedy VRP solver.

    Strategy:
      - Each vehicle starts from the depot.
      - At each step, pick the nearest unvisited customer
        that still fits within the remaining capacity.
      - Return to depot when no more customers fit.
    """
    G = instance.graph
    unvisited = [n for n in G.nodes if n != instance.depot]
    routes = {}
    total_cost = 0.0

    for v in range(instance.n_vehicles):
        vehicle_key = f"vehicle_{v + 1}"
        route = [instance.depot]
        load = 0
        current = instance.depot

        while unvisited:
            # Nearest feasible customer
            feasible = [
                c for c in unvisited
                if load + instance.demands[c] <= instance.capacity
            ]
            if not feasible:
                break

            nearest = min(
                feasible,
                key=lambda c: G[current][c].get("weight", float("inf")),
            )
            route.append(nearest)
            load += instance.demands[nearest]
            total_cost += G[current][nearest].get("weight", 0)
            current = nearest
            unvisited.remove(nearest)

        # Return to depot (only add cost if vehicle actually left the depot)
        route.append(instance.depot)
        if current != instance.depot:
            total_cost += G[current][instance.depot].get("weight", 0)
        routes[vehicle_key] = route

    return {
        "routes": routes,
        "total_cost": round(total_cost, 4),
        "method": "greedy_nearest_neighbour",
        "unserved": unvisited,
    }


def brute_force_vrp(instance: VRPInstance) -> Dict:
    """
    Exhaustive brute-force VRP solver.

    Strategy:
      - Enumerate ALL permutations of customers.
      - For each permutation, greedily split into n_vehicles routes
        (fill each vehicle until capacity is exceeded, then start next one).
      - Track the split assignment that yields the minimum total travel cost.
      - Depot (node 0) is always start/end of every route.

    Complexity: O((n-1)!) — only feasible for n_customers <= 7.
    """
    G = instance.graph
    customers = [n for n in G.nodes if n != instance.depot]

    best_cost = float("inf")
    best_routes: Dict = {}

    t0 = time.perf_counter()

    for perm in itertools.permutations(customers):
        # Split permutation into vehicle routes respecting capacity
        routes: Dict[str, List[int]] = {}
        vehicle_idx = 0
        load = 0
        route: List[int] = [instance.depot]
        total_cost = 0.0
        feasible = True

        for customer in perm:
            demand = instance.demands[customer]
            if vehicle_idx >= instance.n_vehicles:
                feasible = False
                break
            if load + demand > instance.capacity:
                # Close current vehicle route
                route.append(instance.depot)
                total_cost += G[route[-2]][instance.depot].get("weight", 0)
                routes[f"vehicle_{vehicle_idx + 1}"] = route
                vehicle_idx += 1
                if vehicle_idx >= instance.n_vehicles:
                    feasible = False
                    break
                route = [instance.depot]
                load = 0
            prev = route[-1]
            total_cost += G[prev][customer].get("weight", 0)
            route.append(customer)
            load += demand

        if not feasible:
            continue

        # Close last vehicle route
        route.append(instance.depot)
        total_cost += G[route[-2]][instance.depot].get("weight", 0)
        routes[f"vehicle_{vehicle_idx + 1}"] = route

        # Pad remaining vehicles with empty depot→depot routes
        for v in range(vehicle_idx + 2, instance.n_vehicles + 1):
            routes[f"vehicle_{v}"] = [instance.depot, instance.depot]

        if total_cost < best_cost:
            best_cost = total_cost
            best_routes = {k: list(v) for k, v in routes.items()}

    elapsed = time.perf_counter() - t0

    return {
        "routes": best_routes,
        "total_cost": round(best_cost, 4),
        "method": "brute_force",
        "runtime_s": round(elapsed, 6),
        "unserved": [],
    }


def route_cost(G: nx.Graph, route: List[int]) -> float:
    return sum(
        G[route[i]][route[i + 1]].get("weight", 0)
        for i in range(len(route) - 1)
    )


def print_result(result: Dict, instance: VRPInstance) -> None:
    print("\n=== Greedy VRP Result ===")
    for vehicle, route in result["routes"].items():
        cost = route_cost(instance.graph, route)
        load = sum(instance.demands[n] for n in route if n != instance.depot)
        print(f"  {vehicle}: {' -> '.join(map(str, route))}")
        print(f"           cost={round(cost, 4)}  load={load}/{instance.capacity}")
    print(f"  Total cost : {result['total_cost']}")
    print(f"  Unserved   : {result['unserved'] if result['unserved'] else 'None'}")


def draw_vrp_routes(
    instance: VRPInstance,
    result: Dict,
    filename: str = "vrp_classical_routes.png",
) -> None:
    G = instance.graph
    pos = instance.positions
    colors = ["tomato", "mediumseagreen", "dodgerblue", "orange", "purple"]

    plt.figure(figsize=(9, 7))

    # Base graph
    nx.draw_networkx_nodes(
        G, pos,
        node_color=["gold" if n == instance.depot else "lightblue" for n in G.nodes()],
        node_size=1100,
    )
    node_labels = {
        n: f"{n}\n(depot)" if n == instance.depot else f"{n}\nd={instance.demands[n]}"
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=0.5, alpha=0.4)

    # Route edges per vehicle
    legend_handles = []
    for idx, (vehicle, route) in enumerate(result["routes"].items()):
        color = colors[idx % len(colors)]
        edge_list = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_list,
            edge_color=color,
            width=3,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
        )
        cost = route_cost(G, route)
        legend_handles.append(
            mpatches.Patch(color=color,
                           label=f"{vehicle} (cost={round(cost,2)})")
        )

    plt.legend(handles=legend_handles, loc="upper right")
    plt.title(f"Greedy VRP Routes — Total cost: {result['total_cost']}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/{filename}")


if __name__ == "__main__":
    instance = build_vrp_instance(n_customers=4, n_vehicles=2, capacity=10)

    print("\n--- Greedy Nearest-Neighbour ---")
    result_greedy = greedy_vrp(instance)
    print_result(result_greedy, instance)

    print("\n--- Brute Force ---")
    result_bf = brute_force_vrp(instance)
    for veh, route in result_bf["routes"].items():
        c = route_cost(instance.graph, route)
        load = sum(instance.demands[n] for n in route if n != instance.depot)
        print(f"  {veh}: {' -> '.join(map(str, route))}")
        print(f"           cost={round(c, 4)}  load={load}/{instance.capacity}")
    print(f"  Total cost : {result_bf['total_cost']}")
    print(f"  Runtime    : {result_bf['runtime_s']}s")

    draw_vrp_routes(instance, result_bf, filename="vrp_brute_force_routes.png")
