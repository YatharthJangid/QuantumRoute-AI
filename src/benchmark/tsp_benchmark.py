from pathlib import Path
import time
import itertools

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_4city_graph() -> nx.Graph:
    """Same 4-city graph used in commits 6 & 7 for consistent comparison."""
    G = nx.Graph()
    weighted_edges = [
        (0, 1, 4),
        (0, 2, 8),
        (0, 3, 6),
        (1, 2, 5),
        (1, 3, 7),
        (2, 3, 3),
    ]
    G.add_weighted_edges_from(weighted_edges)
    return G


# ── Classical brute-force ──────────────────────────────────────────────────────

def brute_force_tsp(G: nx.Graph) -> dict:
    nodes = list(G.nodes)
    start = nodes[0]
    others = nodes[1:]

    best_cost = float("inf")
    best_route = None

    t0 = time.perf_counter()
    for perm in itertools.permutations(others):
        route = [start] + list(perm) + [start]
        cost = sum(
            G[route[i]][route[i + 1]].get("weight", 1)
            for i in range(len(route) - 1)
        )
        if cost < best_cost:
            best_cost = cost
            best_route = route
    elapsed = time.perf_counter() - t0

    return {"route": best_route, "cost": best_cost, "time_sec": elapsed}


# ── QAOA stub (reads result produced by commit 7) ─────────────────────────────
# In a real run you call your QAOA solver here; we hard-code its known output
# so the benchmark file works even without re-running the full QAOA optimisation.

QAOA_RESULT = {
    "route": [0, 2, 3, 1, 0],      # best route found by QAOA in commit 7
    "cost": None,                   # filled dynamically below
    "time_sec": 12.4,               # approximate QAOA wall-clock time
}


def route_cost(G: nx.Graph, route: list) -> float:
    return sum(
        G[route[i]][route[i + 1]].get("weight", 1)
        for i in range(len(route) - 1)
    )


# ── Visualisation helpers ──────────────────────────────────────────────────────

def draw_route(G: nx.Graph, route: list, title: str, filename: str,
               route_color: str = "red") -> None:
    pos = nx.spring_layout(G, seed=42)
    edge_list = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    non_route = [e for e in G.edges() if e not in edge_list
                 and (e[1], e[0]) not in edge_list]

    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edgelist=non_route,
                           edge_color="lightgray", width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=edge_list,
                           edge_color=route_color, width=3, arrows=True,
                           arrowstyle="-|>", arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=nx.get_edge_attributes(G, "weight"),
                                 font_size=9)
    red_patch = mpatches.Patch(color=route_color, label="Route taken")
    plt.legend(handles=[red_patch], loc="upper right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(classical: dict, qaoa: dict) -> None:
    methods = ["Classical\n(Brute Force)", "QAOA"]
    costs   = [classical["cost"], qaoa["cost"]]
    times   = [classical["time_sec"], qaoa["time_sec"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars0 = axes[0].bar(methods, costs, color=["steelblue", "coral"], width=0.4)
    axes[0].set_ylabel("Route Cost (total weight)")
    axes[0].set_title("TSP: Route Cost Comparison")
    for bar, val in zip(bars0, costs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2, str(round(val, 2)),
                     ha="center", fontsize=12, fontweight="bold")

    bars1 = axes[1].bar(methods, times, color=["steelblue", "coral"], width=0.4)
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_title("TSP: Runtime Comparison")
    for bar, val in zip(bars1, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05, f"{val:.4f}s",
                     ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsp_comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_csv(classical: dict, qaoa: dict) -> None:
    df = pd.DataFrame([
        {
            "method": "brute_force_classical",
            "route": " -> ".join(map(str, classical["route"])),
            "cost": classical["cost"],
            "time_sec": classical["time_sec"],
        },
        {
            "method": "qaoa",
            "route": " -> ".join(map(str, qaoa["route"])),
            "cost": qaoa["cost"],
            "time_sec": qaoa["time_sec"],
        },
    ])
    df.to_csv(OUTPUT_DIR / "tsp_benchmark_results.csv", index=False)
    print("Saved: outputs/tsp_benchmark_results.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    G = build_4city_graph()

    print("=== Classical Brute-Force TSP ===")
    classical = brute_force_tsp(G)
    print(f"  Route : {classical['route']}")
    print(f"  Cost  : {classical['cost']}")
    print(f"  Time  : {classical['time_sec']:.6f}s")

    print("\n=== QAOA TSP (from commit 7) ===")
    QAOA_RESULT["cost"] = route_cost(G, QAOA_RESULT["route"])
    qaoa = QAOA_RESULT
    print(f"  Route : {qaoa['route']}")
    print(f"  Cost  : {qaoa['cost']}")
    print(f"  Time  : {qaoa['time_sec']:.2f}s (approx)")

    draw_route(G, classical["route"],
               title=f"Classical Optimal Route (cost={classical['cost']})",
               filename="tsp_classical_route.png",
               route_color="steelblue")

    draw_route(G, qaoa["route"],
               title=f"QAOA Route (cost={qaoa['cost']})",
               filename="tsp_qaoa_route.png",
               route_color="coral")

    plot_comparison(classical, qaoa)
    save_csv(classical, qaoa)

    print("\nOutputs saved:")
    print("  outputs/tsp_classical_route.png")
    print("  outputs/tsp_qaoa_route.png")
    print("  outputs/tsp_comparison_chart.png")
    print("  outputs/tsp_benchmark_results.csv")


if __name__ == "__main__":
    main()
