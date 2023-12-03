from typing import TypeAlias, List

import numpy as np

from quantum_impl.circuitery import circuit

StateVectorData: TypeAlias = np.ndarray[np.complex64] | List[complex]




def inner_product(vector1: StateVectorData, vector2: StateVectorData) -> float:
    return np.dot(np.conj(vector1), vector2)


def create_circuit_and_project_to_ideal_vector(
    theta_aux: np.ndarray, entanglement: str, ideal_vector: StateVectorData
) -> float:
    c = circuit(theta_aux, entanglement)
    return inner_product(ideal_vector, c.psi)


def calculate_fidelity(
        theta_aux: np.ndarray, entanglement: str, ideal_vector: StateVectorData
) -> float:
    return np.abs(create_circuit_and_project_to_ideal_vector(theta_aux, entanglement, ideal_vector))

