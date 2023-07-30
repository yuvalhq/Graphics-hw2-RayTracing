from typing import Tuple

import numpy as np

from consts import EPSILON


def orthonormal_vector_pair(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Receive a vector and return a pair of two vectors which are orthonormal to it.
    """
    fixed_vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if np.allclose(vector, fixed_vector, atol=EPSILON):
        fixed_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vec1 = np.cross(vector, fixed_vector)
    vec1 /= np.linalg.norm(vec1)
    vec2 = np.cross(vector, vec1)
    vec2 /= np.linalg.norm(vec2)
    return vec1, vec2
