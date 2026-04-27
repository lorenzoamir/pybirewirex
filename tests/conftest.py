"""Shared fixtures for pybirewirex tests."""

import numpy as np
import pytest


@pytest.fixture
def small_bipartite() -> np.ndarray:
    """4×5 binary bipartite incidence matrix."""
    return np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=np.int16,
    )


@pytest.fixture
def small_undirected() -> np.ndarray:
    """5×5 symmetric binary adjacency matrix (no self-loops)."""
    adj = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=np.int16,
    )
    return adj
