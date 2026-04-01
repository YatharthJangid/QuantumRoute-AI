import itertools
import sys
import time
from math import inf
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.vrp_graph import VRPInstance, analyze_vrp_instance, build_vrp_instance


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def _customer_nodes(instance: VRPInstance) -> List[int]:
    return [n for n in sorted(instance.graph.nodes()) if n != instance.depot]


def _empty_routes(instance: VRPInstance) -> Dict[str, List[int]]:
    return {
        f"vehicle_{v + 1}": [instance.depot, instance.depot]
        for v in range(max(instance.n_vehicles, 0))
    }


def route_cost(G: nx.Graph, route: List[int]) -> float:
    return sum(
        G[route[i]][route[i + 1]].get("weight", 0)
        for i in range(len(route) - 1)
        if route[i] != route[i + 1]
    )


def route_load(instance: VRPInstance, route: List[int]) -> int:
    return sum(instance.demands.get(node, 0) for node in route if node != instance.depot)


def _pad_routes(instance: VRPInstance, routes: Dict[str, List[int]]) -> Dict[str, List[int]]:
    padded = {key: list(route) for key, route in routes.items()}
    for vehicle_idx in range(1, instance.n_vehicles + 1):
        padded.setdefault(f"vehicle_{vehicle_idx}", [instance.depot, instance.depot])
    return padded


def _visited_customers(instance: VRPInstance, routes: Dict[str, List[int]]) -> set[int]:
    return {
        node
        for route in routes.values()
        for node in route
        if node != instance.depot
    }


def _finalize_result(
    instance: VRPInstance,
    method: str,
    routes: Dict[str, List[int]],
    runtime_s: float,
    status: str = "ok",
    unserved: List[int] | None = None,
) -> Dict:
    finalized_routes = _pad_routes(instance, routes) if len(routes) <= instance.n_vehicles else {
        key: list(route) for key, route in routes.items()
    }

    if unserved is None:
        visited = _visited_customers(instance, finalized_routes)
        unserved = [node for node in _customer_nodes(instance) if node not in visited]
    else:
        unserved = sorted(unserved)

    total_cost = sum(route_cost(instance.graph, route) for route in finalized_routes.values())
    route_count_ok = len(finalized_routes) <= instance.n_vehicles if instance.n_vehicles >= 0 else False
    feasible = not unserved and route_count_ok and status == "ok"

    if not route_count_ok and status == "ok":
        status = (
            f"incomplete: requires {len(finalized_routes)} routes "
            f"for {instance.n_vehicles} vehicles"
        )
        feasible = False

    return {
        "routes": finalized_routes,
        "total_cost": float(round(total_cost, 4)) if total_cost < inf else inf,
        "method": method,
        "runtime_s": round(runtime_s, 6),
        "unserved": unserved,
        "feasible": feasible,
        "status": status,
    }


def _failure_result(
    instance: VRPInstance,
    method: str,
    runtime_s: float,
    status: str,
    unserved: List[int] | None = None,
) -> Dict:
    if unserved is None:
        unserved = _customer_nodes(instance)
    routes = _empty_routes(instance)
    return {
        "routes": routes,
        "total_cost": inf,
        "method": method,
        "runtime_s": round(runtime_s, 6),
        "unserved": sorted(unserved),
        "feasible": False,
        "status": status,
    }


def greedy_vrp(instance: VRPInstance) -> Dict:
    """
    Nearest-neighbour greedy VRP solver.

    Strategy:
      - Each vehicle starts from the depot.
      - At each step, pick the nearest unvisited customer
        that still fits within the remaining capacity.
      - Return to depot when no more customers fit.
    """
    start = time.perf_counter()
    analysis = analyze_vrp_instance(instance)
    customers = analysis["customers"]

    if not analysis["is_valid"]:
        status = "invalid: " + "; ".join(analysis["issues"])
        return _failure_result(instance, "greedy_nearest_neighbour", time.perf_counter() - start, status)

    if not customers:
        return _finalize_result(
            instance,
            "greedy_nearest_neighbour",
            _empty_routes(instance),
            time.perf_counter() - start,
        )

    if instance.n_vehicles == 0:
        status = "infeasible: no vehicles available to serve customers"
        return _failure_result(instance, "greedy_nearest_neighbour", time.perf_counter() - start, status, customers)

    G = instance.graph
    unvisited = list(customers)
    routes = {}

    for v in range(instance.n_vehicles):
        vehicle_key = f"vehicle_{v + 1}"
        route = [instance.depot]
        load = 0
        current = instance.depot

        while unvisited:
            feasible = [
                customer for customer in unvisited
                if load + instance.demands[customer] <= instance.capacity
            ]
            if not feasible:
                break

            nearest = min(
                feasible,
                key=lambda customer: G[current][customer].get("weight", float("inf")),
            )
            route.append(nearest)
            load += instance.demands[nearest]
            current = nearest
            unvisited.remove(nearest)

        route.append(instance.depot)
        routes[vehicle_key] = route

    status = "ok" if not unvisited else "incomplete: customers remain unserved"
    return _finalize_result(
        instance,
        "greedy_nearest_neighbour",
        routes,
        time.perf_counter() - start,
        status=status,
        unserved=unvisited,
    )


def brute_force_vrp(instance: VRPInstance) -> Dict:
    """
    Exact brute-force VRP solver.

    Strategy:
      - Explore all customer visit orders and all route split points.
      - Start a new vehicle whenever doing so might improve the solution.
      - Track the minimum total travel cost across all feasible assignments.

    Complexity grows explosively; intended only for very small instances.
    """
    start = time.perf_counter()
    analysis = analyze_vrp_instance(instance)
    customers = analysis["customers"]

    if not analysis["is_valid"]:
        status = "invalid: " + "; ".join(analysis["issues"])
        return _failure_result(instance, "brute_force", time.perf_counter() - start, status)

    if not customers:
        return _finalize_result(
            instance,
            "brute_force",
            _empty_routes(instance),
            time.perf_counter() - start,
        )

    if not analysis["is_feasible"]:
        reasons = analysis["infeasible_reasons"] or ["instance is infeasible"]
        status = "infeasible: " + "; ".join(reasons)
        return _failure_result(instance, "brute_force", time.perf_counter() - start, status, customers)

    G = instance.graph
    depot = instance.depot
    best_cost = inf
    best_routes: Dict[str, List[int]] | None = None

    def search(
        remaining: tuple[int, ...],
        vehicle_idx: int,
        current_route: List[int],
        current_load: int,
        current_node: int,
        routes_so_far: List[List[int]],
        cost_so_far: float,
    ) -> None:
        nonlocal best_cost, best_routes

        if cost_so_far >= best_cost:
            return

        if not remaining:
            final_cost = cost_so_far
            final_routes = list(routes_so_far)
            if len(current_route) > 1:
                final_routes.append(current_route + [depot])
                final_cost += G[current_node][depot].get("weight", 0)

            result_routes = {
                f"vehicle_{idx + 1}": route
                for idx, route in enumerate(final_routes)
            }
            result_routes = _pad_routes(instance, result_routes)

            if final_cost < best_cost:
                best_cost = final_cost
                best_routes = {key: list(route) for key, route in result_routes.items()}
            return

        if len(current_route) > 1 and vehicle_idx + 1 < instance.n_vehicles:
            close_cost = cost_so_far + G[current_node][depot].get("weight", 0)
            search(
                remaining,
                vehicle_idx + 1,
                [depot],
                0,
                depot,
                routes_so_far + [current_route + [depot]],
                close_cost,
            )

        for idx, customer in enumerate(remaining):
            demand = instance.demands[customer]
            if current_load + demand > instance.capacity:
                continue

            travel_cost = G[current_node][customer].get("weight", 0)
            next_remaining = remaining[:idx] + remaining[idx + 1:]
            search(
                next_remaining,
                vehicle_idx,
                current_route + [customer],
                current_load + demand,
                customer,
                routes_so_far,
                cost_so_far + travel_cost,
            )

    search(tuple(customers), 0, [depot], 0, depot, [], 0.0)

    if best_routes is None:
        status = "infeasible: no feasible route assignment found"
        return _failure_result(instance, "brute_force", time.perf_counter() - start, status, customers)

    return _finalize_result(
        instance,
        "brute_force",
        best_routes,
        time.perf_counter() - start,
    )


def _is_route_endpoint(node: int, route: List[int]) -> bool:
    return bool(route) and (route[0] == node or route[-1] == node)


def _merge_customer_sequences(
    left_route: List[int],
    left_node: int,
    right_route: List[int],
    right_node: int,
) -> List[int]:
    left = list(left_route)
    right = list(right_route)

    if left[-1] != left_node:
        left.reverse()
    if right[0] != right_node:
        right.reverse()

    return left + right


def _best_sequence_merge(
    route_a: List[int],
    route_b: List[int],
    depot: int,
    G: nx.Graph,
) -> tuple[List[int], float]:
    best_route = None
    best_delta = inf

    candidates_a = [list(route_a), list(reversed(route_a))]
    candidates_b = [list(route_b), list(reversed(route_b))]

    for left in candidates_a:
        for right in candidates_b:
            delta = (
                G[left[-1]][right[0]].get("weight", 0)
                - G[left[-1]][depot].get("weight", 0)
                - G[depot][right[0]].get("weight", 0)
            )
            if delta < best_delta:
                best_delta = delta
                best_route = left + right

    return best_route, best_delta


def clarke_wright_vrp(instance: VRPInstance) -> Dict:
    """
    Clarke-Wright savings heuristic for capacitated VRP.

    Strategy:
      - Start with one route per customer.
      - Compute savings s(i, j) = c(0, i) + c(0, j) - c(i, j).
      - Merge routes in descending savings order whenever the merge keeps both
        endpoints valid and the combined load stays within capacity.
    """
    start = time.perf_counter()
    analysis = analyze_vrp_instance(instance)
    customers = analysis["customers"]

    if not analysis["is_valid"]:
        status = "invalid: " + "; ".join(analysis["issues"])
        return _failure_result(instance, "clarke_wright_savings", time.perf_counter() - start, status)

    if not customers:
        return _finalize_result(
            instance,
            "clarke_wright_savings",
            _empty_routes(instance),
            time.perf_counter() - start,
        )

    if instance.n_vehicles == 0:
        status = "infeasible: no vehicles available to serve customers"
        return _failure_result(instance, "clarke_wright_savings", time.perf_counter() - start, status, customers)

    G = instance.graph
    depot = instance.depot
    unserved = list(analysis["customers_over_capacity"])
    serviceable = [customer for customer in customers if customer not in unserved]

    routes = {customer: [customer] for customer in serviceable}
    route_loads = {customer: instance.demands[customer] for customer in serviceable}
    node_to_route = {customer: customer for customer in serviceable}

    savings = []
    for customer_i, customer_j in itertools.combinations(serviceable, 2):
        saving = (
            G[depot][customer_i].get("weight", 0)
            + G[depot][customer_j].get("weight", 0)
            - G[customer_i][customer_j].get("weight", 0)
        )
        savings.append((saving, customer_i, customer_j))
    savings.sort(reverse=True)

    for _, customer_i, customer_j in savings:
        route_id_i = node_to_route.get(customer_i)
        route_id_j = node_to_route.get(customer_j)
        if route_id_i is None or route_id_j is None or route_id_i == route_id_j:
            continue

        route_i = routes[route_id_i]
        route_j = routes[route_id_j]

        if not _is_route_endpoint(customer_i, route_i) or not _is_route_endpoint(customer_j, route_j):
            continue
        if route_loads[route_id_i] + route_loads[route_id_j] > instance.capacity:
            continue

        merged_route = _merge_customer_sequences(route_i, customer_i, route_j, customer_j)
        routes[route_id_i] = merged_route
        route_loads[route_id_i] += route_loads[route_id_j]

        del routes[route_id_j]
        del route_loads[route_id_j]
        for customer in merged_route:
            node_to_route[customer] = route_id_i

    while len(routes) > instance.n_vehicles:
        best_merge = None
        route_ids = list(routes.keys())
        for route_id_i, route_id_j in itertools.combinations(route_ids, 2):
            if route_loads[route_id_i] + route_loads[route_id_j] > instance.capacity:
                continue

            merged_route, delta = _best_sequence_merge(
                routes[route_id_i],
                routes[route_id_j],
                depot,
                G,
            )
            if best_merge is None or delta < best_merge[0]:
                best_merge = (delta, route_id_i, route_id_j, merged_route)

        if best_merge is None:
            break

        _, route_id_i, route_id_j, merged_route = best_merge
        routes[route_id_i] = merged_route
        route_loads[route_id_i] += route_loads[route_id_j]
        del routes[route_id_j]
        del route_loads[route_id_j]
        for customer in merged_route:
            node_to_route[customer] = route_id_i

    ordered_sequences = sorted(routes.values(), key=lambda seq: tuple(seq))
    result_routes = {
        f"vehicle_{idx + 1}": [depot] + sequence + [depot]
        for idx, sequence in enumerate(ordered_sequences)
    }

    status = "ok"
    if len(result_routes) > instance.n_vehicles:
        status = (
            f"incomplete: heuristic produced {len(result_routes)} routes "
            f"for {instance.n_vehicles} vehicles"
        )
    elif unserved:
        status = "incomplete: some customers exceed vehicle capacity"

    return _finalize_result(
        instance,
        "clarke_wright_savings",
        result_routes,
        time.perf_counter() - start,
        status=status,
        unserved=unserved,
    )


def print_result(result: Dict, instance: VRPInstance) -> None:
    print(f"\n=== {result['method']} Result ===")
    for vehicle, route in result["routes"].items():
        cost = route_cost(instance.graph, route)
        load = route_load(instance, route)
        print(f"  {vehicle}: {' -> '.join(map(str, route))}")
        print(f"           cost={round(cost, 4)}  load={load}/{instance.capacity}")
    print(f"  Total cost : {result['total_cost']}")
    print(f"  Feasible   : {result['feasible']}")
    print(f"  Status     : {result['status']}")
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

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=["gold" if n == instance.depot else "lightblue" for n in G.nodes()],
        node_size=1100,
    )
    node_labels = {
        n: f"{n}\n(depot)" if n == instance.depot else f"{n}\nd={instance.demands[n]}"
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=0.5, alpha=0.4)

    legend_handles = []
    for idx, (vehicle, route) in enumerate(result["routes"].items()):
        color = colors[idx % len(colors)]
        edge_list = [
            (route[i], route[i + 1])
            for i in range(len(route) - 1)
            if route[i] != route[i + 1]
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_list,
            edge_color=color,
            width=3,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
        )
        cost = route_cost(G, route)
        legend_handles.append(
            mpatches.Patch(color=color, label=f"{vehicle} (cost={round(cost, 2)})")
        )

    if legend_handles:
        plt.legend(handles=legend_handles, loc="upper right")
    plt.title(f"{result['method']} - Total cost: {result['total_cost']}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/{filename}")


if __name__ == "__main__":
    instance = build_vrp_instance(n_customers=4, n_vehicles=2, capacity=10)

    print("\n--- Greedy Nearest-Neighbour ---")
    print_result(greedy_vrp(instance), instance)

    print("\n--- Clarke-Wright Savings ---")
    print_result(clarke_wright_vrp(instance), instance)

    print("\n--- Brute Force ---")
    result_bf = brute_force_vrp(instance)
    print_result(result_bf, instance)

    draw_vrp_routes(instance, result_bf, filename="vrp_brute_force_routes.png")
