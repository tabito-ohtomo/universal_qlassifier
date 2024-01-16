##########################################################################
# Quantum classifier
# Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
# Code by APS
# Code-checks by ACL
# June 3rd 2019
from enum import Enum
from typing import Tuple, List, Any, TypeAlias

import numpy as np

from domain.learning import LabeledDataSet, LabeledDataSetToLearn, split_dataset

# Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos
###########################################################################
## This file creates the data points for the different problems to be tackled by the quantum classifier

problems = ['circle', '3 circles', 'wavy circle', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares',
            'wavy lines', 'iris']



class PROBLEM(Enum):
    IRIS = 1

def data_generator(problem:PROBLEM) -> Tuple[LabeledDataSetToLearn, Any]:
    """
    This function generates the data for a problem
    INPUT: 
        -problem: Name of the problem, one of: 'circle', '3 circles', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines'
        -samples Number of samples for the data
    OUTPUT:
        -data: set of training and test data
        -settings: things needed for drawing
    """
    if problem == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        ret: LabeledDataSet = []
        for d, t in zip(data.data, data.target):
            # if t == 0:
            #     ret.append((d, t))
          ret.append((d, t))

        train_data = []
        test_data = []
        train_data.extend(ret[:30])
        train_data.extend(ret[50:80])
        train_data.extend(ret[100:130])
        test_data.extend(ret[30:50])
        test_data.extend(ret[80:100])
        test_data.extend(ret[130:150])
        return (train_data, test_data), ()
