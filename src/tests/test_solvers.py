"""
test_solvers.py - Correctness assertions for the VRP solvers.

Run with:
    venv\Scripts\python.exe -m src.tests.test_solvers
"""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.qaoa_solver import solve_vrp_qaoa
from src.core.vrp_classical import brute_force_vrp, clarke_wright_vrp, greedy_vrp
from src.core.vrp_graph import build_vrp_instance


class SolverTests(unittest.TestCase):
    def assert_routes_feasible(self, result, instance):
        for vehicle, route in result["routes"].items():
            load = sum(instance.demands.get(node, 0) for node in route)
            self.assertLessEqual(
                load,
                instance.capacity,
                msg=f"{vehicle} exceeds capacity with load {load}",
            )

    def assert_serves_all_customers(self, result, instance):
        visited = {
            node
            for route in result["routes"].values()
            for node in route
            if node != instance.depot
        }
        expected = {node for node in instance.graph.nodes if node != instance.depot}
        reported_unserved = set(result.get("unserved", []))
        self.assertEqual(visited | reported_unserved, expected)

    def test_small_instance(self):
        inst = build_vrp_instance(n_customers=4, n_vehicles=2, capacity=15, seed=42)

        brute = brute_force_vrp(inst)
        greedy = greedy_vrp(inst)
        savings = clarke_wright_vrp(inst)

        self.assertTrue(brute["feasible"])
        self.assertTrue(greedy["feasible"])
        self.assertTrue(savings["feasible"])

        self.assert_serves_all_customers(brute, inst)
        self.assert_serves_all_customers(greedy, inst)
        self.assert_serves_all_customers(savings, inst)

        self.assertLessEqual(brute["total_cost"], greedy["total_cost"] + 0.01)
        self.assertLessEqual(brute["total_cost"], savings["total_cost"] + 0.01)

    def test_capacity_constraints(self):
        inst = build_vrp_instance(n_customers=6, n_vehicles=2, capacity=12, seed=7)

        brute = brute_force_vrp(inst)
        greedy = greedy_vrp(inst)
        savings = clarke_wright_vrp(inst)

        self.assert_routes_feasible(brute, inst)
        self.assert_routes_feasible(greedy, inst)
        self.assert_routes_feasible(savings, inst)

    def test_single_customer(self):
        inst = build_vrp_instance(n_customers=1, n_vehicles=2, capacity=10, seed=42)

        brute = brute_force_vrp(inst)
        greedy = greedy_vrp(inst)
        savings = clarke_wright_vrp(inst)

        self.assertGreater(brute["total_cost"], 0)
        self.assertGreater(greedy["total_cost"], 0)
        self.assertGreater(savings["total_cost"], 0)

    def test_zero_customers(self):
        inst = build_vrp_instance(n_customers=0, n_vehicles=2, capacity=10, seed=42)

        brute = brute_force_vrp(inst)
        greedy = greedy_vrp(inst)
        savings = clarke_wright_vrp(inst)
        qaoa = solve_vrp_qaoa(inst, p=1, shots=32, maxiter=1, seed=42)

        for result in (brute, greedy, savings):
            self.assertTrue(result["feasible"])
            self.assertEqual(result["total_cost"], 0.0)
            self.assertEqual(result["unserved"], [])

        self.assertTrue(qaoa["feasible"])
        self.assertEqual(qaoa["best_cost"], 0.0)

    def test_infeasible_instance_handling(self):
        inst = build_vrp_instance(n_customers=4, n_vehicles=1, capacity=3, seed=42)

        brute = brute_force_vrp(inst)
        greedy = greedy_vrp(inst)
        savings = clarke_wright_vrp(inst)

        self.assertFalse(brute["feasible"])
        self.assertEqual(brute["total_cost"], float("inf"))
        self.assertFalse(greedy["feasible"])
        self.assertFalse(savings["feasible"])
        self.assertTrue(greedy["unserved"])
        self.assertTrue(savings["unserved"])

    def test_qaoa_feasibility(self):
        inst = build_vrp_instance(n_customers=4, n_vehicles=2, capacity=15, seed=42)
        result = solve_vrp_qaoa(inst, p=1, shots=128, maxiter=4, seed=42)

        self.assertTrue(result["feasible"])
        self.assertGreater(result["best_cost"], 0)
        self.assertLess(result["best_cost"], float("inf"))
        for vehicle, route in result["routes"].items():
            load = sum(inst.demands.get(node, 0) for node in route)
            self.assertLessEqual(load, inst.capacity, msg=f"{vehicle} exceeds capacity")

    def test_qaoa_requires_exactly_two_vehicles(self):
        inst = build_vrp_instance(n_customers=4, n_vehicles=3, capacity=10, seed=42)
        with self.assertRaises(ValueError):
            solve_vrp_qaoa(inst, p=1, shots=32, maxiter=1, seed=42)

    def test_brute_force_is_best_across_classical_heuristics(self):
        for seed in [1, 7, 13, 42, 99]:
            inst = build_vrp_instance(n_customers=5, n_vehicles=2, capacity=14, seed=seed)
            brute = brute_force_vrp(inst)
            greedy = greedy_vrp(inst)
            savings = clarke_wright_vrp(inst)
            self.assertLessEqual(brute["total_cost"], greedy["total_cost"] + 0.01)
            self.assertLessEqual(brute["total_cost"], savings["total_cost"] + 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
