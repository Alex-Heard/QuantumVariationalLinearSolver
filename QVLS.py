
import numpy as np
from qiskit import transpile
from qiskit_algorithms.optimizers import COBYLA, SLSQP, QNSPSA, SPSA
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, UCC, HartreeFock
from qiskit_aer.primitives import EstimatorV2 as Estimator, SamplerV2 as AerSampler, EstimatorV2 as AerEstimator
from qiskit_aer import Aer
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, EfficientSU2, RZZGate, ExcitationPreserving, EvolvedOperatorAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler, Sampler
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from qiskit.result import marginal_counts
import time
import traceback
from qiskit.quantum_info import SparsePauliOp

A_matrix = np.array([
    [2, 0, 0, 1],
    [0, 2, 1, 0],
    [0, 1, 2, 0],
    [1, 0, 0, 2]
])
pauli_decomposition = SparsePauliOp.from_operator(A_matrix)

qc1 = QuantumCircuit(2)
qc2 = QuantumCircuit(2)
qc3 = QuantumCircuit(2)
qc4 = QuantumCircuit(2)

qc2.x(0)
qc3.x(1)
qc4.x([0,1])

state1 = Statevector(qc1)
state2 = Statevector(qc2)
state3 = Statevector(qc3)
state4 = Statevector(qc4)

states = [state1, state2, state3, state4]

final_states = []

ansatz = EfficientSU2(num_qubits=2, reps=3, entanglement='linear')
num_params = ansatz.num_parameters

for i, s in enumerate(states):
    iteration_count = 0

    def vqls_cost_function(params_values):
        global iteration_count
        iteration_count += 1
        
        # 1. Bind parameters to create our guessed |x> state
        bound_circ = ansatz.assign_parameters(params_values)
        x_state = Statevector(bound_circ)
        
        # 2. Apply the matrix A to our guess (A|x>)
        Ax_state = x_state.evolve(A_matrix)
        
        # 3. Calculate Denominator: <x| A_dagger A |x>
        # (Since A is real and symmetric, A_dagger = A, so this is just the inner product of Ax with itself)
        den = Ax_state.inner(Ax_state).real
        
        # 4. Calculate Numerator: |<b| A |x>|^2
        # How much does A|x> overlap with our target |b>?
        overlap = s.inner(Ax_state)
        num = np.abs(overlap)**2
        
        # 5. The VQLS Cost Formula
        cost = 1.0 - (num / den)
        
        if iteration_count % 50 == 0:
            print(f"  Eval {iteration_count}: Cost = {cost:.6f}")
            
        return cost

    rng = np.random.default_rng(1234)
    initial_guess = rng.uniform(-np.pi, np.pi, num_params)

    print("\nLaunching COBYLA Optimization for VQLS...")

    cobyla = COBYLA(maxiter=1500, tol=1e-5)
    result = cobyla.minimize(fun=vqls_cost_function, x0=initial_guess)

    optimal_circ = ansatz.assign_parameters(result.x)
    vqls_state = Statevector(optimal_circ).data

    classical_x = np.linalg.solve(A_matrix, s.data)
    classical_x_normalized = classical_x / np.linalg.norm(classical_x)
    classical_norm = np.linalg.norm(classical_x)

    if np.sign(vqls_state[0].real) != np.sign(classical_x_normalized[0].real):
        vqls_state = -vqls_state

    print("Classical Exact |x>: ", np.round(classical_x_normalized.real, 4))
    print("Quantum VQLS  |x>: ", np.round(vqls_state.real, 4))

    # Calculate absolute vector error
    error = np.linalg.norm(classical_x_normalized.real - vqls_state.real)
    print(f"\nTotal State Vector Error: {error:.4f}")

    final_states.append(np.abs(vqls_state))



print("\n--- PRESENTATION METRICS ---")
print(f"Total Qubits Used: {ansatz.num_qubits} (Plus 1 Ancilla if run on real hardware)")
print(f"Circuit Depth (Ansatz): {ansatz.decompose().depth()} gates")

#Visualize
A_inverse = np.column_stack((final_states[0], final_states[1], final_states[2], final_states[3]))
np.set_printoptions(precision=4, suppress=True, linewidth=100)
A_inverse = np.real(A_inverse)
A_inverse = A_inverse*classical_norm
# qc1.draw(output='mpl', filename='HHL_Abstract_Circuit.png', style='clifford')
print(A_inverse)

true_hardware_circuit = transpile(optimal_circ, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)

print(f"Abstract Depth: {optimal_circ.depth()}")
print(f"Hardware Depth: {true_hardware_circuit.depth()}")