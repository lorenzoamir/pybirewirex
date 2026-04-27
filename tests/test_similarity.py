import numpy as np
import pytest

from pybirewirex.similarity import jaccard


class TestJaccard:
    def test_identical_matrices(self):
        m = np.array([[1, 0, 1], [0, 1, 0]])
        assert jaccard(m, m) == pytest.approx(1.0)

    def test_disjoint_matrices(self):
        m1 = np.array([[1, 0], [0, 1]])
        m2 = np.array([[0, 1], [1, 0]])
        assert jaccard(m1, m2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # intersection = {(0,0)}, union = {(0,0),(0,1),(1,1)}
        m1 = np.array([[1, 0], [0, 1]])
        m2 = np.array([[1, 1], [0, 0]])
        result = jaccard(m1, m2)
        assert result == pytest.approx(1.0 / 3.0)

    def test_all_zeros(self):
        m = np.zeros((3, 3), dtype=int)
        assert jaccard(m, m) == pytest.approx(1.0)

    def test_symmetric(self):
        m1 = np.array([[1, 0, 1], [0, 1, 1]])
        m2 = np.array([[1, 1, 0], [0, 1, 0]])
        assert jaccard(m1, m2) == pytest.approx(jaccard(m2, m1))

    def test_float_binary_input(self):
        m1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        m2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert jaccard(m1, m2) == pytest.approx(1.0)

    def test_returns_float(self):
        m = np.eye(3, dtype=int)
        assert isinstance(jaccard(m, m), float)
