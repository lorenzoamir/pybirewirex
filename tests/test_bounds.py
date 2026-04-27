import math
from pybirewirex._bounds import bound_bipartite, bound_undirected


# Reference values computed by hand from the R formulas.
# Bipartite: e=10, t=20, accuracy=1e-5
#   one_minus = 0.5, log_term = log(0.5/1e-5) = log(50000)
_LOG50K = math.log(50000)


class TestBoundBipartite:
    def test_standard_known_value(self):
        e, t, acc = 10.0, 20.0, 1e-5
        # ceil((e / (2*(1-e/t))) * log((1-e/t)/acc))
        # = ceil((10 / 1.0) * log(50000))
        expected = math.ceil(10.0 * _LOG50K)
        assert bound_bipartite(e, t, acc, exact=False) == expected

    def test_exact_known_value(self):
        e, t, acc = 10.0, 20.0, 1e-5
        # ceil((e*(1-e/t)) * log((1-e/t)/acc) / 2)
        # = ceil(5.0 * log(50000) / 2)
        expected = math.ceil(5.0 * _LOG50K / 2.0)
        assert bound_bipartite(e, t, acc, exact=True) == expected

    def test_exact_less_than_standard(self):
        # Exact bound should be smaller (stricter) than standard
        val_std = bound_bipartite(10.0, 20.0, 1e-5, exact=False)
        val_exact = bound_bipartite(10.0, 20.0, 1e-5, exact=True)
        assert val_exact < val_std

    def test_returns_int(self):
        result = bound_bipartite(6.0, 24.0, 1e-5, exact=False)
        assert isinstance(result, int)

    def test_higher_accuracy_smaller_bound(self):
        b1 = bound_bipartite(10.0, 20.0, 1e-3, exact=False)
        b2 = bound_bipartite(10.0, 20.0, 1e-5, exact=False)
        assert b1 < b2

    def test_sparse_graph(self):
        # Very sparse: e=1, t=100 → ratio=0.01
        result = bound_bipartite(1.0, 100.0, 1e-5, exact=False)
        assert result > 0


class TestBoundUndirected:
    def test_exact_known_value(self):
        # n=5 nodes, t=12.5, e=4, accuracy=1e-5
        e, t, acc = 4.0, 12.5, 1e-5
        one_minus = 1.0 - e / t  # = 0.68
        log_term = math.log(one_minus / acc)
        expected = math.ceil((e * one_minus * log_term) / 2.0)
        assert bound_undirected(e, t, acc, exact=True) == expected

    def test_standard_known_value(self):
        e, t, acc = 4.0, 12.5, 1e-5
        d = e / t
        denom = 2.0 * d**3 - 6.0 * d**2 + 2.0 * d + 2.0
        log_term = math.log((1.0 - d) / acc)
        expected = math.ceil((e / denom) * log_term)
        assert bound_undirected(e, t, acc, exact=False) == expected

    def test_returns_int(self):
        result = bound_undirected(4.0, 12.5, 1e-5, exact=True)
        assert isinstance(result, int)

    def test_higher_accuracy_smaller_bound(self):
        b1 = bound_undirected(4.0, 12.5, 1e-3, exact=False)
        b2 = bound_undirected(4.0, 12.5, 1e-5, exact=False)
        assert b1 < b2

    def test_positive_result(self):
        for exact in (True, False):
            assert bound_undirected(4.0, 12.5, 1e-5, exact=exact) > 0
