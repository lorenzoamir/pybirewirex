"""Tests for Issue #5: dense undirected rewiring and analysis."""

import numpy as np

from pybirewirex.bipartite import AnalysisResult
from pybirewirex.undirected import analysis_undirected, rewire_undirected


def test_rewire_undirected_preserves_degree_sequence(small_undirected):
    result = rewire_undirected(small_undirected, seed=42, verbose=False)
    np.testing.assert_array_equal(
        np.sort(result.sum(axis=0)), np.sort(small_undirected.sum(axis=0))
    )


def test_rewire_undirected_is_symmetric(small_undirected):
    result = rewire_undirected(small_undirected, seed=42, verbose=False)
    np.testing.assert_array_equal(result, result.T)


def test_rewire_undirected_no_self_loops(small_undirected):
    result = rewire_undirected(small_undirected, seed=42, verbose=False)
    np.testing.assert_array_equal(np.diag(result), 0)


def test_rewire_undirected_returns_ndarray(small_undirected):
    result = rewire_undirected(small_undirected, seed=1, verbose=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == small_undirected.shape


def test_rewire_undirected_deterministic(small_undirected):
    r1 = rewire_undirected(small_undirected, seed=99, verbose=False)
    r2 = rewire_undirected(small_undirected, seed=99, verbose=False)
    np.testing.assert_array_equal(r1, r2)


def test_rewire_undirected_different_seeds_differ(small_undirected):
    r1 = rewire_undirected(small_undirected, seed=1, max_iter=500, verbose=False)
    r2 = rewire_undirected(small_undirected, seed=2, max_iter=500, verbose=False)
    assert not np.array_equal(r1, r2)


def test_rewire_undirected_explicit_max_iter(small_undirected):
    result = rewire_undirected(small_undirected, max_iter=200, seed=7, verbose=False)
    np.testing.assert_array_equal(
        np.sort(result.sum(axis=0)), np.sort(small_undirected.sum(axis=0))
    )


def test_rewire_undirected_exact_true(small_undirected):
    result = rewire_undirected(
        small_undirected, max_iter="n", exact=True, seed=3, verbose=False
    )
    np.testing.assert_array_equal(
        np.sort(result.sum(axis=0)), np.sort(small_undirected.sum(axis=0))
    )


def test_rewire_undirected_exact_false(small_undirected):
    result = rewire_undirected(
        small_undirected, max_iter="n", exact=False, seed=4, verbose=False
    )
    np.testing.assert_array_equal(
        np.sort(result.sum(axis=0)), np.sort(small_undirected.sum(axis=0))
    )


def test_rewire_undirected_verbose_no_error(small_undirected):
    rewire_undirected(small_undirected, max_iter=50, seed=5, verbose=True)


def test_analysis_undirected_returns_analysis_result(small_undirected):
    result = analysis_undirected(
        small_undirected, step=5, n_networks=3, max_iter=20, seed=0, verbose=False
    )
    assert isinstance(result, AnalysisResult)


def test_analysis_undirected_scores_shape(small_undirected):
    n_networks = 4
    step = 5
    max_iter = 20
    result = analysis_undirected(
        small_undirected,
        step=step,
        n_networks=n_networks,
        max_iter=max_iter,
        seed=0,
        verbose=False,
    )
    # n_steps = (20-1)//5 + 2 = 5
    assert result.scores.shape == (n_networks, 5)


def test_analysis_undirected_first_score_is_one(small_undirected):
    result = analysis_undirected(
        small_undirected, step=5, n_networks=5, max_iter=20, seed=10, verbose=False
    )
    np.testing.assert_array_equal(result.scores[:, 0], 1.0)


def test_analysis_undirected_scores_in_unit_interval(small_undirected):
    result = analysis_undirected(
        small_undirected, step=5, n_networks=3, max_iter=20, seed=11, verbose=False
    )
    assert np.all(result.scores >= 0.0)
    assert np.all(result.scores <= 1.0)


def test_analysis_undirected_n_matches_bound(small_undirected):
    from pybirewirex._bounds import bound_undirected

    m = small_undirected
    n = m.shape[0]
    e = int(np.sum(np.triu(m, k=1)))
    t = n * (n - 1) // 2
    expected_N = bound_undirected(e, t, 1e-5, False)

    result = analysis_undirected(
        m, step=5, n_networks=2, max_iter="n", exact=False, seed=0, verbose=False
    )
    assert result.N == expected_N


def test_analysis_undirected_exact_true(small_undirected):
    from pybirewirex._bounds import bound_undirected

    m = small_undirected
    n = m.shape[0]
    e = int(np.sum(np.triu(m, k=1)))
    t = n * (n - 1) // 2
    expected_N = bound_undirected(e, t, 1e-5, True)

    result = analysis_undirected(
        m, step=5, n_networks=2, max_iter="n", exact=True, seed=0, verbose=False
    )
    assert result.N == expected_N


def test_analysis_undirected_step_stored(small_undirected):
    result = analysis_undirected(
        small_undirected, step=7, n_networks=2, max_iter=20, seed=0, verbose=False
    )
    assert result.step == 7


def test_analysis_undirected_verbose_no_error(small_undirected):
    analysis_undirected(
        small_undirected, step=5, n_networks=2, max_iter=20, seed=0, verbose=True
    )
