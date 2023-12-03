##########################################################################
# Quantum classifier
# Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
# Code by APS
# Code-checks by ACL
# June 3rd 2019


# Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

## This file will create the circuit we will use in order to develop the quantum
# classifier. Ansätze and codings are included in this file 

## How this file is related to other files

from quantum_impl.QuantumState import QCircuit


def circuit(theta_aux, entanglement) -> QCircuit:
    """
    This creates the Quantum circuit for the problem using QuantumState.QCircuit
    INPUT: 
        -theta_aux: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3 or 6). Alpha and x are coded inside theta_aux
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        -quantum circuits coding the problem and our Ansätze
    """
    hypar = theta_aux.shape  # [qubits, layers, params_per_layer]
    entanglement = entanglement.lower()[0]
    if hypar[-1] not in [3, 6]:
        raise ValueError('The number of parameters per gate is not correct')

    if hypar[-1] == 3:
        num_qubits = hypar[0]
        if num_qubits == 1:
            return _qcircuit_1qubit(theta_aux)

        elif num_qubits == 2 and entanglement == 'n':
            return _qcircuit_2qubit_noentanglement(theta_aux)

        elif num_qubits == 2 and entanglement == 'y':
            return _qcircuit_2qubit_entanglement(theta_aux)

        elif num_qubits == 4 and entanglement == 'n':
            return _qcircuit_4qubit_noentanglement(theta_aux)

        elif num_qubits == 4 and entanglement == 'y':
            return _qcircuit_4qubit_entanglement(theta_aux)

        else:
            raise ValueError('Not valid')

    if hypar[-1] == 6:
        num_qubits = hypar[0]
        if num_qubits == 1:
            return _double_qcircuit_1qubit(theta_aux)

        elif num_qubits == 2 and entanglement == 'n':
            return _double_qcircuit_2qubit_noentanglement(theta_aux)

        elif num_qubits == 2 and entanglement == 'y':
            return _double_qcircuit_2qubit_entanglement(theta_aux)

        elif num_qubits == 4 and entanglement == 'n':
            return _double_qcircuit_4qubit_noentanglement(theta_aux)

        elif num_qubits == 4 and entanglement == 'y':
            return _double_qcircuit_4qubit_entanglement(theta_aux)

        else:
            raise ValueError('Not valid')

    # Auxiliary functions. Modify these functions for modifying or creating Ansätze


def _qcircuit_1qubit(theta_aux):
    C = QCircuit(1)
    for l in range(theta_aux.shape[1]):
        C.U3(0, theta_aux[0, l, :])
    return C


def _qcircuit_2qubit_noentanglement(theta_aux):
    C = QCircuit(2)
    for l in range(theta_aux.shape[1]):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :])
    return C


def _qcircuit_2qubit_entanglement(theta_aux):
    C = QCircuit(2)
    for l in range(theta_aux.shape[1] - 1):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :])
        C.Cz(0, 1)
    for q in range(theta_aux.shape[0]):
        C.U3(q, theta_aux[q, -1, :])
    return C


def _qcircuit_4qubit_noentanglement(theta_aux):
    C = QCircuit(4)
    for l in range(theta_aux.shape[1]):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :])
    return C


def _qcircuit_4qubit_entanglement(theta_aux):
    C = QCircuit(4)
    for l in range(theta_aux.shape[1] - 1):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :])
        if l % 2 == 0:
            C.Cz(0, 1)
            C.Cz(2, 3)
        elif l % 2 == 1:
            C.Cz(1, 2)
            C.Cz(0, 3)
    for q in range(theta_aux.shape[0]):
        C.U3(q, theta_aux[q, -1, :])
    return C

    # Auxiliary functions. Modify these functions for modifying or creating Ansätze


def _double_qcircuit_1qubit(theta_aux):
    C = QCircuit(1)
    for l in range(theta_aux.shape[1]):
        C.U3(0, theta_aux[0, l, :3])
        C.U3(0, theta_aux[0, l, 3:])
    return C


def _double_qcircuit_2qubit_noentanglement(theta_aux):
    C = QCircuit(2)
    for l in range(theta_aux.shape[1]):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :3])
            C.U3(q, theta_aux[q, l, 3:])
    return C


def _double_qcircuit_2qubit_entanglement(theta_aux):
    C = QCircuit(2)
    for l in range(theta_aux.shape[1] - 1):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :3])
            C.U3(q, theta_aux[q, l, 3:])
        C.Cz(0, 1)
    for q in range(theta_aux.shape[0]):
        C.U3(q, theta_aux[q, -1, :3])
        C.U3(q, theta_aux[q, -1, 3:])
    return C


def _double_qcircuit_4qubit_noentanglement(theta_aux):
    C = QCircuit(4)
    for l in range(theta_aux.shape[1]):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :3])
            C.U3(q, theta_aux[q, l, 3:])
    return C


def _double_qcircuit_4qubit_entanglement(theta_aux):
    C = QCircuit(4)
    for l in range(theta_aux.shape[1] - 1):
        for q in range(theta_aux.shape[0]):
            C.U3(q, theta_aux[q, l, :3])
            C.U3(q, theta_aux[q, l, 3:])
        if l % 2 == 0:
            C.Cz(0, 1)
            C.Cz(2, 3)
        elif l % 2 == 1:
            C.Cz(1, 2)
            C.Cz(0, 3)
    for q in range(theta_aux.shape[0]):
        C.U3(q, theta_aux[q, -1, :3])
        C.U3(q, theta_aux[q, -1, 3:])
    return C
