
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def demo_basic_circuit():
    print("=== Building First Quantum Circuit ===")
    
    # Create a circuit with 2 qubits and 2 classical bits
    qc = QuantumCircuit(2, 2)
    
    # Apply Hadamard (H) gate to qubit 0 (Superposition)
    qc.h(0)
    
    # Apply CNOT gate: control is qubit 0, target is qubit 1 (Entanglement)
    qc.cx(0, 1)
    
    # Measure both qubits
    qc.measure([0, 1], [0, 1])
    
    print("\nCircuit Diagram:")
    print(qc.draw())

    # Simulate the circuit
    print("\n=== Running on AerSimulator ===")
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    
    # Run 1000 times (shots)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    print("Measurement counts (Expected ~50% 00 and ~50% 11):")
    print(counts)

if __name__ == "__main__":
    demo_basic_circuit()
