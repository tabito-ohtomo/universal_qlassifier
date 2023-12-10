from enum import Enum
from typing import Dict, Any

import numpy as np

from domain.learning import LabeledDataSet
from domain.quantum import StateVectorData
from quantum_circuit import inner_product, create_circuit_by_qiskit, code_coords
from quantum_impl.circuitery import circuit


class OPTIMIZATION_QUANUM_IMPL(Enum):
    SALINAS_2020 = 1
    QISKIT = 2


class Context:
    optimization_quantum_impl = OPTIMIZATION_QUANUM_IMPL.SALINAS_2020
    training_data: LabeledDataSet
    test_data: LabeledDataSet
    parameters: Dict[str, np.ndarray]
    hyper_parameters: Dict[str, float]
    parameters_impl_specific: Dict[str, Any]

    def __init__(self):
        pass

    def create_circuit(self, theta, alpha, x):
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            raise NotImplementedError()


    def create_circuit_and_project_to_ideal_vector(
            self, x,
            ideal_vector: StateVectorData
    ) -> float:
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            entanglement = self.parameters_impl_specific['entanglement']
            pass
            # return inner_product(ideal_vector, create_circuit_by_qiskit(theta, alpha, x, entanglement))
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            theta = self.parameters['theta']
            alpha = self.parameters['alpha']
            entanglement = self.parameters_impl_specific['entanglement']
            theta_aux = code_coords(theta, alpha, x)
            c = circuit(theta_aux, entanglement)
            return inner_product(ideal_vector, c.psi)

    def calculate_fidelity(
            self, x,
            # theta_aux: np.ndarray,
            ideal_vector: StateVectorData
    ) -> float:
        # theta_aux = code_coords(theta, alpha, x)
        return np.abs(self.create_circuit_and_project_to_ideal_vector(x, ideal_vector))


    def calculate_averaged_chi_square(self, train_data, repr) -> float:
        chi_square = 0.0
        for x, y in train_data:
            chi_square += (y - self.calculate_fidelity(x, repr)) ** 2
        return chi_square / len(train_data)