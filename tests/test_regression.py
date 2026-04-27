"""Regression tests comparing Python outputs against committed reference data.

Two classes of tests:
1. Python self-regression: re-run C backend with seed=42 and verify outputs are
   bit-for-bit identical to tests/reference_data/*.npy (committed from a known-good
   run). Catches any unintended change to the C code or dispatch logic.

2. R degree-preservation cross-check: load R-generated rewired matrices from
   tests/reference_data/r_*.csv and verify that:
   - Degree sequences match R's inputs exactly (both R and Python must preserve them).
   - Jaccard trajectory first values equal 1.0 (no-rewiring baseline is identical
     regardless of PRNG differences between R and Python).

NOTE: Raw matrix entries differ between R (Mersenne Twister) and Python (xorshift64)
for the same integer seed. Exact entry-level comparison is therefore not attempted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).parent / "reference_data"

SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_npy(name: str) -> np.ndarray:
    return np.load(REF / name)


def _load_r_csv(name: str) -> np.ndarray:
    return np.loadtxt(REF / f"r_{name}.csv", delimiter=",", skiprows=1)


# ---------------------------------------------------------------------------
# Python self-regression: bipartite
# ---------------------------------------------------------------------------


class TestBipartiteSelfRegression:
    def test_rewired_matches_reference(self) -> None:
        from pybirewirex.bipartite import rewire_bipartite

        m = _load_npy("bipartite_input.npy")
        result = rewire_bipartite(m, max_iter=500, verbose=False, seed=SEED)
        reference = _load_npy("bipartite_rewired.npy")
        np.testing.assert_array_equal(result, reference)

    def test_row_sums_preserved(self) -> None:
        m = _load_npy("bipartite_input.npy")
        rewired = _load_npy("bipartite_rewired.npy")
        np.testing.assert_array_equal(m.sum(axis=1), rewired.sum(axis=1))

    def test_col_sums_preserved(self) -> None:
        m = _load_npy("bipartite_input.npy")
        rewired = _load_npy("bipartite_rewired.npy")
        np.testing.assert_array_equal(m.sum(axis=0), rewired.sum(axis=0))

    def test_analysis_scores_match_reference(self) -> None:
        from pybirewirex.bipartite import analysis_bipartite

        m = _load_npy("bipartite_input.npy")
        result = analysis_bipartite(m, step=10, max_iter=500, n_networks=5, verbose=False, seed=SEED)
        reference = _load_npy("bipartite_scores.npy")
        np.testing.assert_allclose(result.scores, reference, rtol=0, atol=1e-15)

    def test_analysis_N_matches_reference(self) -> None:
        from pybirewirex.bipartite import analysis_bipartite

        m = _load_npy("bipartite_input.npy")
        result = analysis_bipartite(m, step=10, max_iter=500, n_networks=5, verbose=False, seed=SEED)
        reference_N = int(_load_npy("bipartite_N.npy")[0])
        assert result.N == reference_N

    def test_analysis_first_scores_are_one(self) -> None:
        scores = _load_npy("bipartite_scores.npy")
        np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-15)

    def test_analysis_scores_in_unit_interval(self) -> None:
        scores = _load_npy("bipartite_scores.npy")
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0 + 1e-15


# ---------------------------------------------------------------------------
# Python self-regression: undirected
# ---------------------------------------------------------------------------


class TestUndirectedSelfRegression:
    def test_rewired_matches_reference(self) -> None:
        from pybirewirex.undirected import rewire_undirected

        m = _load_npy("undirected_input.npy")
        result = rewire_undirected(m, max_iter=500, verbose=False, seed=SEED)
        reference = _load_npy("undirected_rewired.npy")
        np.testing.assert_array_equal(result, reference)

    def test_degree_sequence_preserved(self) -> None:
        m = _load_npy("undirected_input.npy")
        rewired = _load_npy("undirected_rewired.npy")
        np.testing.assert_array_equal(
            np.sort(m.sum(axis=1)), np.sort(rewired.sum(axis=1))
        )

    def test_symmetric(self) -> None:
        rewired = _load_npy("undirected_rewired.npy")
        np.testing.assert_array_equal(rewired, rewired.T)

    def test_no_self_loops(self) -> None:
        rewired = _load_npy("undirected_rewired.npy")
        assert np.all(np.diag(rewired) == 0)

    def test_analysis_scores_match_reference(self) -> None:
        from pybirewirex.undirected import analysis_undirected

        m = _load_npy("undirected_input.npy")
        result = analysis_undirected(m, step=10, max_iter=500, n_networks=5, verbose=False, seed=SEED)
        reference = _load_npy("undirected_scores.npy")
        np.testing.assert_allclose(result.scores, reference, rtol=0, atol=1e-15)

    def test_analysis_first_scores_are_one(self) -> None:
        scores = _load_npy("undirected_scores.npy")
        np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-15)

    def test_analysis_scores_in_unit_interval(self) -> None:
        scores = _load_npy("undirected_scores.npy")
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0 + 1e-15


# ---------------------------------------------------------------------------
# R cross-check: degree preservation
# ---------------------------------------------------------------------------


class TestRCrossCheckBipartite:
    """Verify that Python preserves the same degree structure as R's reference inputs."""

    @pytest.fixture(autouse=True)
    def load_r(self) -> None:
        self.r_input = _load_r_csv("bipartite_input").astype(np.int16)
        self.r_rewired = _load_r_csv("bipartite_rewired").astype(np.int16)

    def test_r_row_sums_preserved(self) -> None:
        np.testing.assert_array_equal(
            self.r_input.sum(axis=1), self.r_rewired.sum(axis=1)
        )

    def test_r_col_sums_preserved(self) -> None:
        np.testing.assert_array_equal(
            self.r_input.sum(axis=0), self.r_rewired.sum(axis=0)
        )

    def test_python_same_row_sums_as_r(self) -> None:
        from pybirewirex.bipartite import rewire_bipartite

        py_rewired = rewire_bipartite(self.r_input, max_iter=500, verbose=False, seed=SEED)
        np.testing.assert_array_equal(
            self.r_input.sum(axis=1), py_rewired.sum(axis=1)
        )

    def test_python_same_col_sums_as_r(self) -> None:
        from pybirewirex.bipartite import rewire_bipartite

        py_rewired = rewire_bipartite(self.r_input, max_iter=500, verbose=False, seed=SEED)
        np.testing.assert_array_equal(
            self.r_input.sum(axis=0), py_rewired.sum(axis=0)
        )

    def test_r_scores_first_col_is_one(self) -> None:
        r_scores = _load_r_csv("bipartite_scores")
        np.testing.assert_allclose(r_scores[:, 0], 1.0, atol=1e-6)

    def test_r_scores_in_unit_interval(self) -> None:
        r_scores = _load_r_csv("bipartite_scores")
        assert float(r_scores.min()) >= 0.0
        assert float(r_scores.max()) <= 1.0 + 1e-6


class TestRCrossCheckUndirected:
    """Verify that Python preserves the same degree structure as R's reference inputs."""

    @pytest.fixture(autouse=True)
    def load_r(self) -> None:
        self.r_input = _load_r_csv("undirected_input").astype(np.int16)
        self.r_rewired = _load_r_csv("undirected_rewired").astype(np.int16)

    def test_r_degree_sequence_preserved(self) -> None:
        np.testing.assert_array_equal(
            np.sort(self.r_input.sum(axis=1)),
            np.sort(self.r_rewired.sum(axis=1)),
        )

    def test_r_rewired_symmetric(self) -> None:
        np.testing.assert_array_equal(self.r_rewired, self.r_rewired.T)

    def test_python_same_degree_sequence_as_r(self) -> None:
        from pybirewirex.undirected import rewire_undirected

        py_rewired = rewire_undirected(self.r_input, max_iter=500, verbose=False, seed=SEED)
        np.testing.assert_array_equal(
            np.sort(self.r_input.sum(axis=1)),
            np.sort(py_rewired.sum(axis=1)),
        )

    def test_r_scores_first_col_is_one(self) -> None:
        r_scores = _load_r_csv("undirected_scores")
        np.testing.assert_allclose(r_scores[:, 0], 1.0, atol=1e-6)

    def test_r_scores_in_unit_interval(self) -> None:
        r_scores = _load_r_csv("undirected_scores")
        assert float(r_scores.min()) >= 0.0
        assert float(r_scores.max()) <= 1.0 + 1e-6
