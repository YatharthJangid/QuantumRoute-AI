from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_noise_model() -> NoiseModel:
    noise_model = NoiseModel()

    error_1q = depolarizing_error(0.01, 1)
    error_2q = depolarizing_error(0.03, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, ["h"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model


def run_counts(circuit: QuantumCircuit, backend: AerSimulator, shots: int = 1024) -> dict:
    compiled = transpile(circuit, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    return result.get_counts()


def main() -> None:
    bell = build_bell_circuit()

    ideal_backend = AerSimulator()
    noisy_backend = AerSimulator(noise_model=build_noise_model())

    ideal_counts = run_counts(bell, ideal_backend)
    noisy_counts = run_counts(bell, noisy_backend)

    print("Ideal counts:", ideal_counts)
    print("Noisy counts:", noisy_counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_histogram(ideal_counts, ax=axes[0], title="Ideal Bell State")
    plot_histogram(noisy_counts, ax=axes[1], title="Noisy Bell State")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "noise_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: outputs/noise_comparison.png")


if __name__ == "__main__":
    main()
