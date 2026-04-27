import numpy as np


def jaccard(m1: np.ndarray, m2: np.ndarray) -> float:
    """Jaccard similarity between two binary matrices.

    Returns |intersection| / |union|. Returns 1.0 if both matrices are all-zero.
    """
    b1 = m1.astype(bool)
    b2 = m2.astype(bool)
    intersection = np.count_nonzero(b1 & b2)
    union = np.count_nonzero(b1 | b2)
    if union == 0:
        return 1.0
    return intersection / union
