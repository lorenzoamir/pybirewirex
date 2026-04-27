"""Tests for Issue #4: dense bipartite rewiring and analysis."""

import numpy as np

from pybirewirex.bipartite import AnalysisResult, analysis_bipartite, rewire_bipartite


def test_rewire_bipartite_preserves_row_sums(small_bipartite):
    result = rewire_bipartite(small_bipartite, seed=42, verbose=False)
    np.testing.assert_array_equal(result.sum(axis=1), small_bipartite.sum(axis=1))


def test_rewire_bipartite_preserves_col_sums(small_bipartite):
    result = rewire_bipartite(small_bipartite, seed=42, verbose=False)
    np.testing.assert_array_equal(result.sum(axis=0), small_bipartite.sum(axis=0))


def test_rewire_bipartite_returns_ndarray(small_bipartite):
    result = rewire_bipartite(small_bipartite, seed=1, verbose=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == small_bipartite.shape


def test_rewire_bipartite_deterministic(small_bipartite):
    r1 = rewire_bipartite(small_bipartite, seed=99, verbose=False)
    r2 = rewire_bipartite(small_bipartite, seed=99, verbose=False)
    np.testing.assert_array_equal(r1, r2)


def test_rewire_bipartite_different_seeds_differ(small_bipartite):
    r1 = rewire_bipartite(small_bipartite, seed=1, max_iter=500, verbose=False)
    r2 = rewire_bipartite(small_bipartite, seed=2, max_iter=500, verbose=False)
    # Different seeds should produce different results with high probability
    assert not np.array_equal(r1, r2)


def test_rewire_bipartite_explicit_max_iter(small_bipartite):
    result = rewire_bipartite(small_bipartite, max_iter=200, seed=7, verbose=False)
    np.testing.assert_array_equal(result.sum(axis=1), small_bipartite.sum(axis=1))
    np.testing.assert_array_equal(result.sum(axis=0), small_bipartite.sum(axis=0))


def test_rewire_bipartite_exact_true(small_bipartite):
    result = rewire_bipartite(
        small_bipartite, max_iter="n", exact=True, seed=3, verbose=False
    )
    np.testing.assert_array_equal(result.sum(axis=1), small_bipartite.sum(axis=1))
    np.testing.assert_array_equal(result.sum(axis=0), small_bipartite.sum(axis=0))


def test_rewire_bipartite_exact_false(small_bipartite):
    result = rewire_bipartite(
        small_bipartite, max_iter="n", exact=False, seed=4, verbose=False
    )
    np.testing.assert_array_equal(result.sum(axis=1), small_bipartite.sum(axis=1))
    np.testing.assert_array_equal(result.sum(axis=0), small_bipartite.sum(axis=0))


def test_rewire_bipartite_verbose_no_error(small_bipartite, capsys):
    rewire_bipartite(small_bipartite, max_iter=50, seed=5, verbose=True)
    # Just confirm it ran without raising


def test_analysis_bipartite_returns_analysis_result(small_bipartite):
    result = analysis_bipartite(
        small_bipartite, step=5, n_networks=3, max_iter=20, seed=0, verbose=False
    )
    assert isinstance(result, AnalysisResult)


def test_analysis_bipartite_scores_shape(small_bipartite):
    n_networks = 4
    step = 5
    max_iter = 20
    result = analysis_bipartite(
        small_bipartite,
        step=step,
        n_networks=n_networks,
        max_iter=max_iter,
        seed=0,
        verbose=False,
    )
    # n_steps = (20-1)//5 + 2 = 3+2 = 5
    assert result.scores.shape == (n_networks, 5)


def test_analysis_bipartite_first_score_is_one(small_bipartite):
    result = analysis_bipartite(
        small_bipartite, step=5, n_networks=5, max_iter=20, seed=10, verbose=False
    )
    np.testing.assert_array_equal(result.scores[:, 0], 1.0)


def test_analysis_bipartite_scores_in_unit_interval(small_bipartite):
    result = analysis_bipartite(
        small_bipartite, step=5, n_networks=3, max_iter=20, seed=11, verbose=False
    )
    assert np.all(result.scores >= 0.0)
    assert np.all(result.scores <= 1.0)


def test_analysis_bipartite_n_matches_bound(small_bipartite):
    from pybirewirex._bounds import bound_bipartite

    m = small_bipartite
    e = int(np.sum(m == 1))
    t = m.shape[0] * m.shape[1]
    expected_N = bound_bipartite(e, t, 1e-5, False)

    result = analysis_bipartite(
        m, step=5, n_networks=2, max_iter="n", exact=False, seed=0, verbose=False
    )
    assert result.N == expected_N


def test_analysis_bipartite_exact_true(small_bipartite):
    from pybirewirex._bounds import bound_bipartite

    m = small_bipartite
    e = int(np.sum(m == 1))
    t = m.shape[0] * m.shape[1]
    expected_N = bound_bipartite(e, t, 1e-5, True)

    result = analysis_bipartite(
        m, step=5, n_networks=2, max_iter="n", exact=True, seed=0, verbose=False
    )
    assert result.N == expected_N


def test_analysis_bipartite_step_stored(small_bipartite):
    result = analysis_bipartite(
        small_bipartite, step=7, n_networks=2, max_iter=20, seed=0, verbose=False
    )
    assert result.step == 7


def test_analysis_bipartite_verbose_no_error(small_bipartite):
    analysis_bipartite(
        small_bipartite, step=5, n_networks=2, max_iter=20, seed=0, verbose=True
    )
