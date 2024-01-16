##########################################################################
# Quantum classifier
# Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
# Code by APS
# Code-checks by ACL
# June 3rd 2019


# Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

# This file is a file taking many different functions from other files and mixing them all together
# so that the usage is automatized

import numpy as np

from data.data_gen import data_generator
from fidelity_minimization import fidelity_minimization
from quantum_optimization_context import create_quantum_context, OPTIMIZATION_QUANUM_IMPL, PROBLEM
from test_data import tester


def minimizer(problem, qubits, entanglement, layers, method, name,
              seed=30, epochs=3000, batch_size=20, eta=0.1):
    """
    This function creates data and minimizes whichever problem (from the selected ones) 
    INPUT:
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -name: a name we want for our our files to be save with
        -seed: seed of numpy.random, needed for replicating results
        -epochs: number of epochs for a 'SGD' method. If there is another method, this input has got no importance
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -weight.txt: saves the weights as a flat array if they exist
    """
    np.random.seed(seed)
    (train_data, test_data), _ = data_generator(problem)

    quantum_context = create_quantum_context(
        # optimization_quantum_impl=OPTIMIZATION_QUANUM_IMPL.SALINAS_2020,
        optimization_quantum_impl=OPTIMIZATION_QUANUM_IMPL.CAPPELETTI_2020,
        problem=PROBLEM.IRIS,
        raw_training_data=train_data,
        raw_test_data=test_data,
        parameters={},  # initialize later
        hyper_parameters={},  # initialize later
        parameters_impl_specific={'entanglement': entanglement},
        parameter_optimization={'chi': 'fidelity_chi',
                                'problem': problem,
                                'method': method},
        ideal_vector={},  # initialize later
    )
    quantum_context.initialize_parameters(qubits, layers)

    # theta, alpha, f = fidelity_minimization(quantum_context, theta, alpha, train_data, reprs,
    f = fidelity_minimization(quantum_context)
    # batch_size, eta, epochs)
    print('==================================== train ====================================')
    (acc_train, acc_train) = tester(quantum_context, quantum_context.training_data)
    quantum_context.original_to_actual_accuracy_table_train = acc_train
    print('==================================== test ====================================')
    (acc_test, acc_map_test) = tester(quantum_context, quantum_context.test_data)
    quantum_context.original_to_actual_accuracy_table_acc_test = acc_test

    quantum_context.write_summary(acc_train, acc_test, f, seed, epochs)
