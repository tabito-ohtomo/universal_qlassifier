##########################################################################
# Quantum classifier
# Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
# Code by APS
# Code-checks by ACL
# June 3rd 2019
import random

import numpy as np
from scipy.optimize import minimize

from domain.learning import LabeledDataSet, LabeledData
from quantum_circuit import create_circuit_and_project_to_ideal_vector
from quantum_optimization_context import QuantumContext
from test_data import tester


# This file provides the minimization for the cheap chi square


# Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos
###########################################################################


def fidelity_minimization(quantum_context: QuantumContext) -> float:
    # batch_size, eta, epochs) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    This function takes the parameters of a problem and computes the optimal parameters for it, using different functions. It uses the fidelity minimization
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -train_data: set of data for training. There must be several entries (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
        -epochs: number of epochs , only for 'SGD' method
    OUTPUT:
        -theta: optimized point for the theta parameters. The shape is correct (qubits, layers, 3)
        -alpha: optimized point for the alpha parameters. The shape is correct (qubits, layers, dim)
        -chi: value of the minimization function
    """

    params = quantum_context.translate_parameters_to_scipy()
    results = minimize(_objective_function_in_scipy, params,
                       args=quantum_context,
                       method=quantum_context.parameter_optimization['method'],
                       options = {'disp': True, 'maxiter':100_000_000})
    optimized_params = results['x']
    optimized_objective_function_value = results['fun']
    quantum_context.kick_back_parameters_from_scipy_params(optimized_params)
    return optimized_objective_function_value


def _gradient(theta, alpha, data, reprs, entanglement):
    """
    This function computes a gradient step for the SGD minimization
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'

    OUTPUT:
        -grad_theta: gradient for the theta parameters. The shape is correct (qubits, layers, 3)
        -grad_alpha: gradient for the alpha parameters. The shape is correct (qubits, layers, dim)
        -results['fun']: value of the minimization function
    """

    x, y = data
    theta_aux = code_coords(theta, alpha, x)
    prod1 = create_circuit_and_project_to_ideal_vector(theta_aux, entanglement, reprs[y])
    prods2 = np.zeros(theta.shape, dtype='complex')
    (Q, L, I) = theta_aux.shape

    for q in range(Q):
        for l in range(L):
            for i in range(I):
                theta_aux[q, l, i] += np.pi
                prods2[q, l, i] = create_circuit_and_project_to_ideal_vector(theta_aux, entanglement, reprs[y])
                theta_aux[q, l, i] -= np.pi
    grad_theta = np.asfarray(np.real(prod1 * prods2))
    if len(x) <= 3:
        dim = len(x)
        grad_alpha = np.empty((theta.shape[0], theta.shape[1], dim))
        for q in range(Q):
            for l in range(L):
                for i in range(dim):
                    grad_alpha[q, l, i] = x[i] * grad_theta[q, l, i]

    if len(x) == 4:
        grad_alpha = np.empty((theta.shape[0], theta.shape[1], 4))
        for q in range(Q):
            grad_alpha[q, l, 0] = x[0] * grad_theta[q, l, 0]
            grad_alpha[q, l, 1] = x[1] * grad_theta[q, l, 1]
            grad_alpha[q, l, 2] = x[2] * grad_theta[q, l, 3]
            grad_alpha[q, l, 3] = x[3] * grad_theta[q, l, 4]

    return grad_theta, grad_alpha


def _train_batch(theta, alpha, batch, reprs, entanglement):
    """
    This function computes a gradient step for a complete batch for the SGD minimization
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -batch: small set of data for training. It must be several (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'

    OUTPUT:
        -grad_theta: gradient for the theta parameters averaged in batch. The shape is correct (qubits, layers, 3)
        -grad_alpha: gradient for the alpha parameters averaged in batch. The shape is correct (qubits, layers, dim)
    """
    gradient_theta = np.zeros(theta.shape)
    gradient_alpha = np.zeros(alpha.shape)
    for d in batch:
        g_t, g_a = _gradient(theta, alpha, d, reprs, entanglement)
        gradient_theta += g_t
        gradient_alpha += g_a

    return gradient_theta / len(batch), gradient_alpha / len(batch)


def _session_sgd(
        quantum_context: QuantumContext,
        theta, alpha, train_data, reprs, entanglement, eta, batch_size):
    """
    This function computes a gradient descent step for all batches
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -train_data: set of data for training. There must be several entries (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -eta: learning rate, only for 'SGD' method
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method

    OUTPUT:
        -theta: updated point for the theta parameters. The shape is correct (qubits, layers, 3)
        -alpha: updated point for the alpha parameters. The shape is correct (qubits, layers, dim)
        -Av_chi_square: value of the minimization function
    """
    batches = [train_data[k:k + batch_size] for k in range(0,
                                                           len(train_data), batch_size)]
    for batch in batches:
        gradient_theta_batch, gradient_alpha_batch = _train_batch(
            theta, alpha, batch, reprs, entanglement)
        theta += eta * gradient_theta_batch  # This sign is very important, it is the difference between maximizing or minimizing.
        alpha += eta * gradient_alpha_batch

    return theta, alpha, av_chi_square(quantum_context)


def _sgd(theta, alpha, train_data, reprs, entanglement, eta, batch_size, epochs):
    """
    This function completes the whole SGD strategy
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -train_data: set of data for training. There must be several entries (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
        -epochs: number of epochs , only for 'SGD' method
    OUTPUT:
        -thetas: optimized points for the theta parameters for all epochs. The shape is correct (qubits, layers, 3)
        -alphas: optimized points for the theta parameters for all epochs. The shape is correct (qubits, layers, dim)
        -chis: value of the minimization function at every step
    """
    thetas = [np.empty(theta.shape)] * epochs
    alphas = [np.empty(alpha.shape)] * epochs
    chis = [0] * epochs
    for e in range(epochs):
        random.shuffle(train_data)
        theta_, alpha_, chi_ = _session_sgd(theta, alpha, train_data, reprs,
                                            entanglement, eta, batch_size)
        thetas[e] = theta_
        alphas[e] = alpha_
        chis[e] = chi_  # Storage for solution

        theta = theta_
        alpha = alpha_  # Next step initialization

    return thetas, alphas, chis



def _objective_function_in_scipy(
        params,
        quantum_context: QuantumContext):
    """
    This function returns the chi^2 function for using scipy
    INPUT:
        -params: theta and alpha inside the same variable
        -hypars: hyperparameters needed to rebuild theta and alpha
        -train_data: training dataset for the classifier
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        - -Av_chi_square, which is the function we want to minimize
    """
    # theta, alpha = _translate_from_scipy(params, hypars)
    quantum_context.kick_back_parameters_from_scipy_params(params)
    current_accuracy = tester(quantum_context, quantum_context.test_data)[0]
    quantum_context.learning_state.add_accuracy(current_accuracy)
    return -av_chi_square(quantum_context)

def av_chi_square(quantum_context: QuantumContext):  # Chi in average
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -Averaged chi^2 for data
    """
    return np.average(quantum_context.calculate_fidelity_batch(quantum_context.training_data))
    # return np.average(list(map(lambda d: _chi_square(quantum_context, d), quantum_context.training_data)))


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
            # elif len(x) == 4:
            #     theta_aux[q, l, 0] += alpha[q, l, 0] * x[0]
            #     theta_aux[q, l, 1] += alpha[q, l, 1] * x[1]
            #     theta_aux[q, l, 2] += alpha[q, l, 2] * x[2]
            #     theta_aux[q, l, 3] += alpha[q, l, 3] * x[3]
            else:
                raise ValueError('Data has too many dimensions')

    return theta_aux
