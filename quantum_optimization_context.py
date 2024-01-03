from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Tuple

import numpy as np

from domain.learning import LabeledDataSet, Label, Data
from domain.quantum import StateVectorData
from quantum_impl.circuitery import circuit
from save_data import create_folder, name_folder


class OPTIMIZATION_QUANUM_IMPL(Enum):
    SALINAS_2020 = 1
    QISKIT = 2

class PROBLEM(Enum):
    IRIS = 1

@dataclass
class QuantumContext:
    optimization_quantum_impl = OPTIMIZATION_QUANUM_IMPL.SALINAS_2020
    problem = PROBLEM.IRIS
    training_data: LabeledDataSet
    test_data: LabeledDataSet
    parameters: Dict[str, np.ndarray]
    hyper_parameters: Dict[str, float]
    parameters_impl_specific: Dict[str, Any]
    parameter_optimization: Dict[str, Any]
    ideal_vector: Dict[Label, StateVectorData]
    #
    # def create_circuit(self, theta, alpha, x):
    #     if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
    #         raise NotImplementedError()

    def initialize_parameters(self, qubits: int, layers: int):
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            self.hyper_parameters['qubits'] = qubits
            self.hyper_parameters['layers'] = layers
            if self.problem == PROBLEM.IRIS:
                self.parameters['theta'] = np.random.rand(qubits, layers, 6)
                self.parameters['alpha'] = np.random.rand(qubits, layers, 4)
                self.hyper_parameters['dim'] = 4
                self.ideal_vector[0] = np.array([1, 0, 0, 0])
                self.ideal_vector[1] = np.array([0, 1, 0, 0])
                self.ideal_vector[2] = np.array([0, 0, 1, 0])


    def translate_parameters_to_scipy(self) -> np.ndarray[float]:
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            return np.concatenate((
                self.parameters['theta'].flatten(),
                self.parameters['alpha'].flatten()))

    def kick_back_parameters_from_scipy_params(self, scipy_params):
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            qubits = self.hyper_parameters['qubits']
            layers = self.hyper_parameters['layers']
            dim = self.hyper_parameters['dim']
            if dim <= 3:
                self.parameters['theta'] = scipy_params[:qubits * layers * 3].reshape(qubits, layers, 3)
                self.parameters['alpha'] = scipy_params[qubits * layers * 3: qubits * layers * 3 + qubits * layers * dim].reshape(qubits, layers, dim)
            else: # dim == 4
                self.parameters['theta'] = scipy_params[:qubits * layers * 6].reshape(qubits, layers, 6)
                self.parameters['alpha'] = scipy_params[qubits * layers * 6: qubits * layers * 6 + qubits * layers * dim].reshape(qubits, layers, dim)

    def translate_hyper_parameters_to_scipy(self) -> Tuple[float, float, float]:  # -> Tuple[int]
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            return self.hyper_parameters['qubits'], self.hyper_parameters['layers'], self.hyper_parameters['dim']

    def get_most_matched_label(self, x: Data) -> Label:
        return max(self.ideal_vector.keys(), key=lambda label: self.calculate_fidelity(x, label))

    def calculate_fidelity(
            self, x: Data,
            label: Label
            # ideal_vector: StateVectorData
    ) -> float:
        # theta_aux = code_coords(theta, alpha, x)
        return np.abs(self.create_circuit_and_project_to_ideal_vector(x, self.ideal_vector[label]))

    def create_circuit_and_project_to_ideal_vector(
            self, x: Data,
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
            # print(ideal_vector)
            # print(c.psi)
            return inner_product(ideal_vector, c.psi)

    def write_summary(self, acc_train, acc_test, chi_value, seed=30, epochs=3000):
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            foldname = name_folder(self.parameter_optimization['chi'],
                                   self.parameter_optimization['problem'],
                                   self.hyper_parameters['qubits'],
                                   self.parameters_impl_specific['entanglement'],
                                   self.hyper_parameters['layers'],
                                   self.parameter_optimization['method'])
            create_folder(foldname)
            file_text = open(foldname + '/' + 'run' + '_summary.txt', 'w')
            file_text.write('\nFigur of merit = ' + self.parameter_optimization['chi'])
            file_text.write('\nProblem = ' + self.parameter_optimization['problem'])
            file_text.write('\nNumber of qubits = ' + str(self.hyper_parameters['qubits']))
            if self.hyper_parameters['qubits'] != 1:
                file_text.write('\nEntanglement = ' + self.parameters_impl_specific['entanglement'])
            file_text.write('\nNumber of layers = ' + str(self.hyper_parameters['layers']))
            file_text.write('\nMinimization method = ' + self.parameter_optimization['method'])
            file_text.write('\nRandom seed = ' + str(seed))
            if self.parameter_optimization['method'] == 'SGD':
                file_text.write('\nNumber of epochs = ' + str(epochs))
            file_text.write('\n\nBEST RESULT\n\n')
            file_text.write('\nTHETA = \n')
            file_text.write(str(self.parameters['theta']))
            file_text.write('\nALPHA = \n')
            file_text.write(str(self.parameters['alpha']))
            # if chi == 'weighted_fidelity_chi':
            #     file_text.write('\nWEIGHTS = \n')
            #     file_text.write(str(weights))
            file_text.write('\nchi**2 = ' + str(chi_value))
            file_text.write('\nacc_train = ' + str(acc_train))
            file_text.write('\nacc_test = ' + str(acc_test))
            file_text.close()


def code_coords(theta: np.ndarray[np.ndarray[np.ndarray[int]]], alpha: np.ndarray[np.ndarray[np.ndarray[int]]],
                x: List[float])\
        -> np.ndarray[np.ndarray[np.ndarray[int]]]:  # Encoding of coordinates
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


def inner_product(vector1: StateVectorData, vector2: StateVectorData) -> float:
    return np.dot(np.conj(vector1), vector2)
