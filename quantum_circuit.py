from typing import TypeAlias, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_impl.circuitery import circuit

StateVectorData: TypeAlias = np.ndarray[np.complex64] | List[complex]


QUANTUM_IMPL = 'qiskit'

def inner_product(vector1: StateVectorData, vector2: StateVectorData) -> float:
    return np.dot(np.conj(vector1), vector2)


def create_circuit_and_project_to_ideal_vector(
        theta, alpha, x,
    # theta_aux: np.ndarray,
        entanglement: str, ideal_vector: StateVectorData
) -> float:
    if QUANTUM_IMPL == 'qiskit':
        return inner_product(ideal_vector, create_circuit_by_qiskit(theta, alpha, x, entanglement))
    elif QUANTUM_IMPL == 'implmented':
        theta_aux = code_coords(theta, alpha, x)
        c = circuit(theta_aux, entanglement)
        return inner_product(ideal_vector, c.psi)



def create_circuit_by_qiskit(
        theta, alpha, x,
        # theta_aux: np.ndarray,
        entanglement: str):
    # hypar = theta_aux.shape  # [qubits, layers, params_per_layer]
    num_qubits = 2
    c = QuantumCircuit(num_qubits)
    ## -----
    mean_0 = 5.893333333333333
    mean_1 = 3.0577777777777775
    mean_2 = 3.803333333333334
    mean_3 = 1.202222222222222

    sigma_0 = 0.844432748457014
    sigma_1 = 0.43664828912416653
    sigma_2 = 1.7857740307465808
    sigma_3 = 0.7526217550488689

    ## -----
    c.x(0)
    phi = (1- alpha[0, 0, 0]/2.0) * np.pi / 2.0 * (x[0] - mean_0) / sigma_0
    c.rz(phi, 0)
    c.x(0)
    ## -----
    c.x(1)
    phi = (1- alpha[1, 0, 0]/2.0) * np.pi / 2.0 * (x[0] - mean_1) / sigma_1
    c.rz(phi, 1)
    c.x(1)
    c.cz(0, 1)

    ## -----
    c.x(0)
    phi = theta[0, 0, 0]
    c.rz(phi, 0)
    c.x(0)
    ## -----
    c.x(1)
    phi = theta[1, 0, 0]
    c.rz(phi, 1)
    c.x(1)
    c.cz(0, 1)


    ## -----
    c.x(0)
    phi = (1- alpha[0, 0, 0]/2.0) * np.pi / 2.0 * (x[0] - mean_2) / sigma_2
    c.rz(phi, 0)
    c.x(0)
    ## -----
    phi = (1- alpha[0, 0, 0]/2.0) * np.pi / 2.0 * (x[0] - mean_3) / sigma_3
    c.x(1)
    c.rz(phi, 1)
    c.x(1)
    c.cz(0, 1)


    ## -----
    c.x(0)
    phi = theta[0, 1, 0]
    c.rz(phi, 0)
    c.x(0)
    ## -----
    c.x(1)
    phi = theta[1, 1, 0]
    c.rz(phi, 1)
    c.x(1)
    c.cz(0, 1)

    # c.snapshot('my_sv',snapshot_type='statevector')
    # service = QiskitRuntimeService()
    # backend = service.backend('simulator_statevector')
    backend = Aer.get_backend('statevector_simulator')
    # backend = Aer.get_backend('simulator_statevector')
    results = execute(c, backend, shots=1).result()
    statevec = results.data()['statevector']
    print(num_qubits)
    print(statevec)
    print(statevec[0])
    return statevec.data



def calculate_fidelity(
        theta, alpha, x,
        # theta_aux: np.ndarray,
        entanglement: str, ideal_vector: StateVectorData
) -> float:
    # theta_aux = code_coords(theta, alpha, x)
    return np.abs(create_circuit_and_project_to_ideal_vector(theta, alpha, x, entanglement, ideal_vector))


def code_coords(theta, alpha, x):  # Encoding of coordinates
    """
    This functions converts theta, alpha and x in a new set of variables encoding the three of them properly
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -x: one data for training, only the coordinates
    OUTPUT:
        -theta_aux: shifted thetas encoding alpha and x inside. Same shape as theta
    """
    theta_aux = theta.copy()
    qubits = theta.shape[0]
    layers = theta.shape[1]
    for q in range(qubits):
        for l in range(layers):
            if len(x) <= 3:
                for i in range(len(x)):
                    theta_aux[q, l, i] += alpha[q, l, i] * x[i]
            elif len(x) == 4:
                theta_aux[q, l, 0] += alpha[q, l, 0] * x[0]
                theta_aux[q, l, 1] += alpha[q, l, 1] * x[1]
                theta_aux[q, l, 2] += alpha[q, l, 2] * x[2]
                theta_aux[q, l, 3] += alpha[q, l, 3] * x[3]
            else:
                raise ValueError('Data has too many dimensions')

    return theta_aux
