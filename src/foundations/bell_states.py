from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram


SIMULATOR = AerSimulator()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_counts(circuit: QuantumCircuit, shots: int = 1024) -> dict:
    compiled = transpile(circuit, SIMULATOR)
    job = SIMULATOR.run(compiled, shots=shots)
    result = job.result()
    return result.get_counts()


def build_phi_plus() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_phi_minus() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.z(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_psi_plus() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def build_psi_minus() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.x(1)
    qc.h(0)
    qc.z(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def save_circuit_diagram(circuit: QuantumCircuit, filename: str) -> None:
    fig = circuit.draw(output="mpl")
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bell_circuits = {
        "phi_plus": build_phi_plus(),
        "phi_minus": build_phi_minus(),
        "psi_plus": build_psi_plus(),
        "psi_minus": build_psi_minus(),
    }

    histogram_data = {}

    for name, circuit in bell_circuits.items():
        counts = run_counts(circuit)
        histogram_data[name] = counts

        print(f"\n{name}")
        print(circuit.draw())
        print("counts:", counts)

        save_circuit_diagram(circuit, f"{name}_circuit.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    titles = {
        "phi_plus": "Phi+",
        "phi_minus": "Phi-",
        "psi_plus": "Psi+",
        "psi_minus": "Psi-",
    }

    for ax, key in zip(axes, histogram_data.keys()):
        plot_histogram(histogram_data[key], ax=ax, title=titles[key])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bell_states_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved files in outputs/:")
    print("- phi_plus_circuit.png")
    print("- phi_minus_circuit.png")
    print("- psi_plus_circuit.png")
    print("- psi_minus_circuit.png")
    print("- bell_states_histograms.png")


if __name__ == "__main__":
    main()
