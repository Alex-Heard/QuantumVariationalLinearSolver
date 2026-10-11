# Variational Quantum Linear Solver (VQLS) & Matrix Inversion

A complete implementation of the **Variational Quantum Linear Solver (VQLS)** algorithm built in Python using **Qiskit**. Designed for NISQ-era quantum processors, this project solves the linear system $A\vert{}x\rangle \propto \vert{}b\rangle$ by pairing a parameterized quantum ansatz with classical optimization, tomographically reconstructing the matrix inverse $A^{-1}$ column by column.

---

## Technical Overview

Unlike fault-tolerant approaches like HHL, VQLS leverages shallow parameterized quantum circuits to minimize gate depth and coherence requirements:

- **Target Matrix & Pauli Decomposition:** Encodes a symmetric $4 \times 4$ linear system matrix $A$ represented directly and mapped via `SparsePauliOp`.
- **Parameterized Ansatz:** Uses an `EfficientSU2` variational form with 2 qubits, linear entanglement topology, and 3 repetition layers ($reps=3$) to generate candidate solutions $\vert{}x(\vec{\theta})\rangle$.
- **Cost Function Formulation:** Optimizes parameters against the normalized projection cost:
  $$C(\vec{\theta}) = 1 - \frac{\vert{}\langle b \vert{} A \vert{} x(\vec{\theta}) \rangle\vert{}^2}{\langle x(\vec{\theta}) \vert{} A^\dagger A \vert{} x(\vec{\theta}) \rangle}$$
  which reaches zero when $\vert{}x(\vec{\theta})\rangle$ is collinear with the true solution $A^{-1}\vert{}b\rangle$.
- **Classical Optimization:** Employs the gradient-free `COBYLA` optimizer over up to 1,500 iterations to iteratively minimize the cost function.
- **Tomographic Inversion Recovery:** Solves four orthogonal basis inputs ($\vert{}00\rangle, \vert{}01\rangle, \vert{}10\rangle, \vert{}11\rangle$) independently, scaling and stacking the resulting normalized state vectors to reconstruct the full inverse matrix $A^{-1}$.
- **Hardware Transpilation Analysis:** Transpiles the optimal circuit to a physical basis gate set (`cx`, `rz`, `sx`, `x`) to evaluate NISQ-viable gate depths.

---

## Circuit & System Specifications

- **Active Qubits:** 2 qubits (ansatz execution)
- **Ansatz Architecture:** `EfficientSU2` (linear entanglement, 3 layers)
- **Optimizer:** `COBYLA` (tolerance = $10^{-5}$, max iterations = 1500)
- **Target Matrix Size:** $4 \times 4$

---

## Prerequisites

```bash
pip install qiskit qiskit-aer qiskit-algorithms numpy matplotlib
