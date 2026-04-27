"""Tests for Issue #7: Pure-NumPy fallback when _C_AVAILABLE = False."""

from __future__ import annotations

import numpy as np
import pytest

import pybirewirex.bipartite as _bp_mod
import pybirewirex.sparse as _sp_mod
import pybirewirex.undirected as _ud_mod
from pybirewirex.bipartite import AnalysisResult, analysis_bipartite, rewire_bipartite
from pybirewirex.undirected import analysis_undirected, rewire_undirected

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bip() -> np.ndarray:
    return np.array(
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]],
        dtype=np.int16,
    )


@pytest.fixture
def adj() -> np.ndarray:
    return np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=np.int16,
    )


@pytest.fixture
def no_c(monkeypatch: pytest.MonkeyPatch):
    """Force _C_AVAILABLE = False in all dispatch modules."""
    monkeypatch.setattr(_bp_mod, "_C_AVAILABLE", False)
    monkeypatch.setattr(_ud_mod, "_C_AVAILABLE", False)
    monkeypatch.setattr(_sp_mod, "_C_AVAILABLE", False)


# ---------------------------------------------------------------------------
# Dense bipartite fallback
# ---------------------------------------------------------------------------


def test_fallback_rewire_bipartite_row_sums(no_c, bip):
    result = rewire_bipartite(bip, seed=42, verbose=False)
    np.testing.assert_array_equal(result.sum(axis=1), bip.sum(axis=1))


def test_fallback_rewire_bipartite_col_sums(no_c, bip):
    result = rewire_bipartite(bip, seed=42, verbose=False)
    np.testing.assert_array_equal(result.sum(axis=0), bip.sum(axis=0))


def test_fallback_rewire_bipartite_returns_ndarray(no_c, bip):
    result = rewire_bipartite(bip, seed=1, verbose=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == bip.shape


def test_fallback_rewire_bipartite_deterministic(no_c, bip):
    r1 = rewire_bipartite(bip, seed=99, verbose=False)
    r2 = rewire_bipartite(bip, seed=99, verbose=False)
    np.testing.assert_array_equal(r1, r2)


def test_fallback_analysis_bipartite_returns_result(no_c, bip):
    result = analysis_bipartite(
        bip, step=5, n_networks=3, max_iter=20, seed=0, verbose=False
    )
    assert isinstance(result, AnalysisResult)


def test_fallback_analysis_bipartite_scores_shape(no_c, bip):
    result = analysis_bipartite(
        bip, step=5, n_networks=4, max_iter=20, seed=0, verbose=False
    )
    assert result.scores.shape == (4, 5)


def test_fallback_analysis_bipartite_first_score_is_one(no_c, bip):
    result = analysis_bipartite(
        bip, step=5, n_networks=3, max_iter=20, seed=0, verbose=False
    )
    np.testing.assert_array_equal(result.scores[:, 0], 1.0)


def test_fallback_analysis_bipartite_scores_in_unit_interval(no_c, bip):
    result = analysis_bipartite(
        bip, step=5, n_networks=3, max_iter=20, seed=0, verbose=False
    )
    assert np.all(result.scores >= 0.0)
    assert np.all(result.scores <= 1.0)


# ---------------------------------------------------------------------------
# Dense undirected fallback
# ---------------------------------------------------------------------------


def test_fallback_rewire_undirected_degree_sequence(no_c, adj):
    result = rewire_undirected(adj, seed=42, verbose=False)
    np.testing.assert_array_equal(np.sort(result.sum(axis=1)), np.sort(adj.sum(axis=1)))


def test_fallback_rewire_undirected_symmetric(no_c, adj):
    result = rewire_undirected(adj, seed=7, verbose=False)
    np.testing.assert_array_equal(result, result.T)


def test_fallback_rewire_undirected_no_self_loops(no_c, adj):
    result = rewire_undirected(adj, seed=3, verbose=False)
    assert np.all(np.diag(result) == 0)


def test_fallback_rewire_undirected_deterministic(no_c, adj):
    r1 = rewire_undirected(adj, seed=55, verbose=False)
    r2 = rewire_undirected(adj, seed=55, verbose=False)
    np.testing.assert_array_equal(r1, r2)


def test_fallback_analysis_undirected_first_score_is_one(no_c, adj):
    result = analysis_undirected(
        adj, step=5, n_networks=3, max_iter=20, seed=0, verbose=False
    )
    np.testing.assert_array_equal(result.scores[:, 0], 1.0)


def test_fallback_analysis_undirected_returns_result(no_c, adj):
    result = analysis_undirected(
        adj, step=5, n_networks=2, max_iter=20, seed=0, verbose=False
    )
    assert isinstance(result, AnalysisResult)


def test_fallback_analysis_undirected_scores_in_unit_interval(no_c, adj):
    result = analysis_undirected(
        adj, step=5, n_networks=3, max_iter=20, seed=0, verbose=False
    )
    assert np.all(result.scores >= 0.0)
    assert np.all(result.scores <= 1.0)


# ---------------------------------------------------------------------------
# Sparse bipartite fallback
# ---------------------------------------------------------------------------


def test_fallback_sparse_bipartite_scipy_row_sums(no_c, bip):
    import scipy.sparse as sp

    sparse_in = sp.csr_matrix(bip)
    from pybirewirex.sparse import rewire_bipartite_sparse

    result = rewire_bipartite_sparse(sparse_in, seed=42, verbose=False)
    dense = np.asarray(result.todense(), dtype=np.int16)
    np.testing.assert_array_equal(dense.sum(axis=1), bip.sum(axis=1))


def test_fallback_sparse_bipartite_scipy_col_sums(no_c, bip):
    import scipy.sparse as sp

    sparse_in = sp.csr_matrix(bip)
    from pybirewirex.sparse import rewire_bipartite_sparse

    result = rewire_bipartite_sparse(sparse_in, seed=42, verbose=False)
    dense = np.asarray(result.todense(), dtype=np.int16)
    np.testing.assert_array_equal(dense.sum(axis=0), bip.sum(axis=0))


def test_fallback_sparse_bipartite_output_type(no_c, bip):
    import scipy.sparse as sp

    sparse_in = sp.csr_matrix(bip)
    from pybirewirex.sparse import rewire_bipartite_sparse

    result = rewire_bipartite_sparse(sparse_in, seed=1, verbose=False)
    assert sp.issparse(result)


# ---------------------------------------------------------------------------
# Sparse undirected fallback
# ---------------------------------------------------------------------------


def test_fallback_sparse_undirected_scipy_degree(no_c, adj):
    import scipy.sparse as sp

    sparse_in = sp.csr_matrix(adj)
    from pybirewirex.sparse import rewire_undirected_sparse

    result = rewire_undirected_sparse(sparse_in, seed=42, verbose=False)
    dense = np.asarray(result.todense(), dtype=np.int16)
    np.testing.assert_array_equal(
        np.sort(dense.sum(axis=1).ravel()), np.sort(adj.sum(axis=1).ravel())
    )


def test_fallback_sparse_undirected_output_type(no_c, adj):
    import scipy.sparse as sp

    sparse_in = sp.csr_matrix(adj)
    from pybirewirex.sparse import rewire_undirected_sparse

    result = rewire_undirected_sparse(sparse_in, seed=1, verbose=False)
    assert sp.issparse(result)
