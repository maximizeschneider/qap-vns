import numpy as np

def hamming_distance(permutation_1: np.ndarray, permutation_2: np.ndarray) -> int:
    """Calculates the hamming distance (number of positions where the elements differ)."""
    return np.sum(permutation_1 != permutation_2)
    