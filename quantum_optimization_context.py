from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Tuple

import numpy as np

from domain.learning import LabeledDataSet
from domain.quantum import StateVectorData
from quantum_circuit import inner_product, create_circuit_by_qiskit
from quantum_impl.circuitery import circuit


class OPTIMIZATION_QUANUM_IMPL(Enum):
    SALINAS_2020 = 1
    QISKIT = 2


@dataclass
class QuantumContext:
    optimization_quantum_impl = OPTIMIZATION_QUANUM_IMPL.SALINAS_2020
    training_data: LabeledDataSet
    test_data: LabeledDataSet
    parameters: Dict[str, np.ndarray]
    hyper_parameters: Dict[str, float]
    parameters_impl_specific: Dict[str, Any]
    #
    # def create_circuit(self, theta, alpha, x):
    #     if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
    #         raise NotImplementedError()

    def translate_parameters_to_scipy(self) -> np.ndarray[float]:
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            return np.concatenate((
                self.parameters['theta'].flatten(),
                self.parameters['alpha'].flatten()))

    def translate_hyper_parameters_to_scipy(self) -> Tuple[float, float, float]:  # -> Tuple[int]
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            return self.hyper_parameters['qubits'], self.hyper_parameters['layers'], self.hyper_parameters['dim']

    def create_circuit_and_project_to_ideal_vector(
            self, x,
            ideal_vector: StateVectorData
    ) -> float:
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            entanglement = self.parameters_impl_specific['entanglement']
            pass
            # return inner_product(ideal_vector, create_circuit_by_qiskit(theta, alpha, x, entanglement))
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
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


def code_coords(theta: np.array[int, int, int], alpha: np.array[int, int. int], x: List[List[float]])\
        -> np.array[int, int, int]:  # Encoding of coordinates
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
