import json
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler
from qiskit.primitives import BackendSampler
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Tuple

import numpy as np
import qiskit.quantum_info as qi
from qiskit import BasicAer
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister

from data.data_gen import PROBLEM
from domain.learning import LabeledDataSet, Label, Data, AccuracyTable, LabeledData
from domain.quantum import StateVectorData
from quantum_impl.circuitery import circuit
from save_data import create_folder, name_folder

II = qi.Pauli('II')
IZ = qi.Pauli('IZ')
ZI = qi.Pauli('ZI')
ZZ = qi.Pauli('ZZ')

minus_IZ = qi.Pauli('-IZ')
minus_ZI = qi.Pauli('-ZI')
minus_ZZ = qi.Pauli('-ZZ')

class OPTIMIZATION_QUANUM_IMPL(Enum):
    SALINAS_2020 = 1
    QISKIT = 2
    CAPPELETTI_2020 = 3


@dataclass
class QuantumContext:
    optimization_quantum_impl: OPTIMIZATION_QUANUM_IMPL
    problem: PROBLEM.IRIS
    training_data: LabeledDataSet
    test_data: LabeledDataSet
    parameters: Dict[str, np.ndarray]
    hyper_parameters: Dict[str, Any]
    parameters_impl_specific: Dict[str, Any]
    parameter_optimization: Dict[str, Any]
    ideal_vector: Dict[Label, StateVectorData]
    original_to_actual_accuracy_table_train: AccuracyTable
    original_to_actual_accuracy_table_test: AccuracyTable

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
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020:
            self.parameters['theta'] = np.random.randn(8)

            # ====================== backends selection ======================
            # provider = IBMProvider()
            # print(provider.backends())
            service = QiskitRuntimeService()
            print(service.backends())
            backend = BasicAer.get_backend("qasm_simulator")

            self.hyper_parameters['backend'] = backend
            # self.hyper_parameters['backend_cloud'] = service.get_backend("ibmq_qasm_simulator")
            # ================================================================
            # self.hyper_parameters['backend'] = # TODO move to sub class field
            if self.problem == PROBLEM.IRIS:
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
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020:
            return self.parameters['theta']

    def kick_back_parameters_from_scipy_params(self, scipy_params):
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            qubits = self.hyper_parameters['qubits']
            layers = self.hyper_parameters['layers']
            dim = self.hyper_parameters['dim']
            if dim <= 3:
                self.parameters['theta'] = scipy_params[:qubits * layers * 3].reshape(qubits, layers, 3)
                self.parameters['alpha'] = scipy_params[
                                           qubits * layers * 3: qubits * layers * 3 + qubits * layers * dim].reshape(
                    qubits, layers, dim)
            else:  # dim == 4
                self.parameters['theta'] = scipy_params[:qubits * layers * 6].reshape(qubits, layers, 6)
                self.parameters['alpha'] = (
                    scipy_params[qubits * layers * 6: qubits * layers * 6 + qubits * layers * dim].reshape(
                        qubits, layers, dim))
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020:
            self.parameters['theta'] = scipy_params

    def predict(self, x: Data) -> Label:
        return max(self.ideal_vector.keys(), key=lambda label: self.calculate_fidelity(x, label))

    def calculate_fidelity_batch(self, labeled_data_set: LabeledDataSet) -> List[float]:
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020:
            theta = self.parameters['theta']

            circuits = list(map(lambda x: self.circuit(theta, x[0]), labeled_data_set))

            # sampler = Sampler(self.hyper_parameters['backend_cloud'])
            sampler = BackendSampler(self.hyper_parameters['backend'])
            job = sampler.run(circuits=circuits)
            result = job.result()
            print(result)
            return list(map(lambda dist_to_data: dist_to_data[0].get(dist_to_data[1][1], 0.0),
                            zip(result.quasi_dists, labeled_data_set)))


    def circuit(self, theta: List[float] | np.ndarray, x: Data) -> QuantumCircuit:
            registers = QuantumRegister(2, 'qr')
            c = QuantumCircuit(registers)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(x[:2], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(theta[:2], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(x[2:4], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(theta[2:4], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(x[:2], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(theta[4:6], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(x[2:4], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)

            c.rx(np.pi / 2, registers)
            for omega, register in zip(theta[6:8], registers):
                c.rz(omega, register)
            c.rx(np.pi / 2, registers)
            c.cz(0, 1)
            c.measure_all()
            return c

    def calculate_fidelity(
            self, x: Data,
            label: Label
    ) -> float:
        if self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.QISKIT:
            pass
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.SALINAS_2020:
            ideal_vector = self.ideal_vector[label]
            theta = self.parameters['theta']
            alpha = self.parameters['alpha']
            entanglement = self.parameters_impl_specific['entanglement']
            theta_aux = code_coords(theta, alpha, x)
            c = circuit(theta_aux, entanglement)
            return inner_product(ideal_vector, c.psi)
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020:
            theta = self.parameters['theta']
            registers = QuantumRegister(2, 'qr')
            c = self.circuit(theta, x)

            sampler = BackendSampler(self.hyper_parameters['backend'])
            job = sampler.run(circuits=c)
            result = job.result()
            return result.quasi_dists[0].get(label, 0.0)

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
        elif self.optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020:
            # foldname = name_folder(self.parameter_optimization['chi'],
            #                        self.parameter_optimization['problem'],
            #                        self.hyper_parameters['qubits'],
            #                        self.parameters_impl_specific['entanglement'],
            #                        self.hyper_parameters['layers'],
            #                        self.parameter_optimization['method'])
            foldname = str(self.parameter_optimization['chi']) + '/' + \
                       str(self.parameter_optimization['problem']) + '/' + \
                       str(self.parameter_optimization['method'])
            create_folder(foldname)
            file_text = open(foldname + '/' + 'run' + '_summary.txt', 'w')
            file_text.write('\nCAPPELETTI_2020')
            file_text.write('\nFigur of merit = ' + self.parameter_optimization['chi'])
            file_text.write('\nProblem = ' + self.parameter_optimization['problem'])
            # file_text.write('\nNumber of qubits = ' + str(self.hyper_parameters['qubits']))
            # if self.hyper_parameters['qubits'] != 1:
            #     file_text.write('\nEntanglement = ' + self.parameters_impl_specific['entanglement'])
            # file_text.write('\nNumber of layers = ' + str(self.hyper_parameters['layers']))
            file_text.write('\nMinimization method = ' + self.parameter_optimization['method'])
            file_text.write('\nRandom seed = ' + str(seed))
            if self.parameter_optimization['method'] == 'SGD':
                file_text.write('\nNumber of epochs = ' + str(epochs))
            file_text.write('\n\nBEST RESULT\n\n')
            file_text.write('\nTHETA = \n')
            file_text.write(str(self.parameters['theta']))
            # file_text.write('\nALPHA = \n')
            # file_text.write(str(self.parameters['alpha']))

            # if chi == 'weighted_fidelity_chi':
            #     file_text.write('\nWEIGHTS = \n')
            #     file_text.write(str(weights))
            file_text.write('\nchi**2 = ' + str(chi_value))
            file_text.write('\nacc_train = ' + str(acc_train))
            file_text.write('\nacc_test = ' + str(acc_test))

            file_text.write('\nacc_train_map: \n')
            file_text.write(str(self.original_to_actual_accuracy_table_train.expected_to_actual))

            file_text.write('\nacc_test_map: \n')
            file_text.write(str(self.original_to_actual_accuracy_table_test.expected_to_actual))

            file_text.close()


def code_coords(theta: np.ndarray[np.ndarray[np.ndarray[int]]], alpha: np.ndarray[np.ndarray[np.ndarray[int]]],
                x: List[float]) \
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


def create_quantum_context(
        optimization_quantum_impl: OPTIMIZATION_QUANUM_IMPL,
        problem: PROBLEM,
        raw_training_data: LabeledDataSet,
        raw_test_data: LabeledDataSet,
        parameters: Dict[str, np.ndarray],
        hyper_parameters: Dict[str, float],
        parameters_impl_specific: Dict[str, Any],
        parameter_optimization: Dict[str, Any],
        ideal_vector: Dict[Label, StateVectorData],
) -> QuantumContext:
    training_data = raw_training_data
    test_data = raw_test_data
    if optimization_quantum_impl == OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020:
        raw_train_data = list(map(lambda labeled_data: labeled_data[0], raw_training_data))
        mean = np.mean(raw_train_data, axis=0)
        std = np.std(raw_train_data, axis=0)
        training_data = transform_zipped_data(lambda x: (x - mean) / std / 3 * 0.95 * np.pi, raw_training_data, index=0)
        test_data = transform_zipped_data(lambda x: (x - mean) / std / 3 * 0.95 * np.pi, raw_test_data, index=0)
    else:
        training_data = raw_training_data
        test_data = raw_test_data

    return QuantumContext(
        optimization_quantum_impl=optimization_quantum_impl,
        problem=problem,
        training_data=training_data,
        test_data=test_data,
        parameters=parameters,
        hyper_parameters=hyper_parameters,
        parameters_impl_specific=parameters_impl_specific,
        parameter_optimization=parameter_optimization,
        ideal_vector=ideal_vector,
        original_to_actual_accuracy_table_train=AccuracyTable(),
        original_to_actual_accuracy_table_test=AccuracyTable())


def inner_product(vector1: StateVectorData, vector2: StateVectorData) -> float:
    return np.dot(np.conj(vector1), vector2)


def transform_zipped_data(func: Callable[[Any], Any], zipped_list: List[Tuple[Any, Any]], index: int) \
        -> List[Tuple[Any, Any]]:
    list_0 = map(lambda x: x[0], zipped_list)
    list_1 = map(lambda x: x[1], zipped_list)

    if index == 0:
        return list(zip(list(map(func, list_0)), list_1))
    elif index == 1:
        return list(zip(list_0, list(map(func, list_1))))
    if index != 0 or index != 1:
        raise Exception('oops!:' + str(index))
