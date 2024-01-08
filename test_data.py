##########################################################################
# Quantum classifier
# Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
# Code by APS
# Code-checks by ACL
# June 3rd 2019
from typing import Tuple, Dict

# Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################


# This file provides useful tools checking how good our results are

import numpy as np

from domain.learning import LabeledDataSet, Label
from fidelity_minimization import code_coords
from quantum_optimization_context import QuantumContext
from weighted_fidelity_minimization import mat_fidelities, w_fidelities


def _claim(
        quantum_context: QuantumContext,
        x, reprs):
    """
    This function takes the parameters of a solved problem and one data computes classification of this point
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -y_: the class of x, according to the classifier
    """
    # chi = chi.lower().replace(' ', '_')
    # if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    # if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
    #     raise ValueError('Figure of merit is not valid')
    #
    # if chi == 'fidelity_chi':
    y_ = _claim_fidelity(quantum_context, x, reprs)

    # if chi == 'weighted_fidelity_chi':
    #     y_ = _claim_weighted_fidelity(quantum_context, theta, alpha, weight, x, reprs, entanglement)

    return y_


def _claim_fidelity(
        quantum_context: QuantumContext,
        x, reprs):
    """
    This function is inside _claim for fidelity_chi
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        the class of x, according to the classifier
    """
    # theta_aux = code_coords(theta, alpha, x)
    fidelities = [quantum_context.calculate_fidelity(x, r) for r in reprs]

    return np.argmax(fidelities)


def _claim_weighted_fidelity(theta, alpha, weight, x, reprs, entanglement):
    """
    This function is inside _claim for weighted_fidelity_chi
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -x: coordinates of data for testing.
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        the class of x, according to the classifier
    """
    theta_aux = code_coords(theta, alpha, x)
    fids = mat_fidelities(theta_aux, weight, reprs, entanglement)
    w_fid = w_fidelities(fids, weight)
    return np.argmax(w_fid)


def tester(quantum_context: QuantumContext):
    """
    This function takes the parameters of a solved problem and one data computes how many points are correct
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -test_data: set of data for testing
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -success normalized
    """
    total_map: Dict[Label, int] = {}
    acc_map: Dict[Label, int] = {}
    for label in set(map(lambda x: x[1], quantum_context.test_data)):
        total_map[label] = 0
        acc_map[label] = 0
    acc = 0
    for d in quantum_context.test_data:
        x, y = d
        y_ = quantum_context.predict(x)
        total_map[y] = total_map[y] + 1
        if y == y_:
            acc += 1
            acc_map[y] = acc_map[y] + 1

        print('expected: ' + str(y))
        print('actual: ' + str(y_))

    print(total_map)
    print(acc_map)

    return acc / len(quantum_context.test_data)


def Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi, weights=None):
    """
    This function takes the parameters of a solved problem and one data computes how many points are correct
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -test_data: set of data for testing
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
    OUTPUT:
        -solutions of the classification
        -success normalized
    """
    dim = len(test_data[0][0])
    solutions = np.zeros((len(test_data), dim + 3))  # data  #Esto se podrá mejorar en el futuro
    for i, d in enumerate(test_data):
        x, y = d
        y_ = _claim(theta, x, reprs)
        solutions[i, :dim] = x
        solutions[i, -3] = y
        solutions[i, -2] = y_
        solutions[i, -1] = int(y == y_)

    acc = np.sum(solutions[:, -1]) / (i + 1)

    return solutions, acc
