import itertools
import time
import networkx as nx
import numpy as np


def brute_force_tsp(G: nx.Graph) -> dict:
    nodes = list(G.nodes)
    start = nodes[0]
    others = nodes[1:]

    best_cost = float("inf")
    best_route = None
    t0 = time.time()

    for perm in itertools.permutations(others):
        route = [start] + list(perm) + [start]
        cost = sum(
            G[route[i]][route[i + 1]].get("weight", 1)
            for i in range(len(route) - 1)
        )
        if cost < best_cost:
            best_cost = cost
            best_route = route

    elapsed = time.time() - t0
    return {
        "route": best_route,
        "cost": best_cost,
        "time_sec": elapsed,
        "method": "brute_force",
    }


def main() -> None:
    G = nx.Graph()
    G.add_weighted_edges_from(
        [
            (0, 1, 10),
            (0, 2, 15),
            (0, 3, 20),
            (1, 2, 35),
            (1, 3, 25),
            (2, 3, 30),
        ]
    )

    result = brute_force_tsp(G)
    print("Best TSP route (classical brute force):", result["route"])
    print("Cost:", result["cost"])
    print("Time (seconds):", result["time_sec"])


if __name__ == "__main__":
    main()
