##########################################################################
# Quantum classifier
# Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
# Code by APS
# Code-checks by ACL
# June 3rd 2019
from typing import Tuple, Dict

from domain.learning import LabeledDataSet, Label, AccuracyTable
from quantum_optimization_context import QuantumContext


# Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos
###########################################################################
# This file provides useful tools checking how good our results are


def tester(quantum_context: QuantumContext, data_to_test: LabeledDataSet) -> Tuple[float, AccuracyTable]:
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
    acc_map: AccuracyTable = AccuracyTable()
    labels = quantum_context.predict_batch(list(map(lambda data: data[0], data_to_test)))
    for (data, expected_label), predicted_label in zip(data_to_test, labels):
        acc_map.add(expected_label, predicted_label)

    return acc_map.get_accuracy(), acc_map
