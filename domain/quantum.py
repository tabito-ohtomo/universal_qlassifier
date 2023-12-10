from typing import TypeAlias, List

import numpy as np

StateVectorData: TypeAlias = np.ndarray[np.complex64] | List[complex]
