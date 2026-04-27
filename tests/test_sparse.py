"""Tests for sparse and graph input dispatch (issue #6)."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pybirewirex import rewire_bipartite, rewire_undirected


# ---- Fixtures ----


@pytest.fixture
def bip_matrix() -> np.ndarray:
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
def bip_sparse(bip_matrix):
    return sp.csr_matrix(bip_matrix)


@pytest.fixture
def bip_igraph(bip_matrix):
    igraph = pytest.importorskip("igraph")
    nrow, ncol = bip_matrix.shape
    n = nrow + ncol
    # Vertices 0..nrow-1 are row partition (type=False),
    # vertices nrow..nrow+ncol-1 are col partition (type=True).
    types = [False] * nrow + [True] * ncol
    edges = [
        (i, nrow + j)
        for i in range(nrow)
        for j in range(ncol)
        if bip_matrix[i, j] == 1
    ]
    G = igraph.Graph(n=n, edges=edges, directed=False)
    G.vs["type"] = types
    return G


@pytest.fixture
def bip_nx(bip_matrix):
    nx = pytest.importorskip("networkx")
    nrow, ncol = bip_matrix.shape
    G = nx.Graph()
    G.add_nodes_from(range(nrow), bipartite=0)
    G.add_nodes_from(range(nrow, nrow + ncol), bipartite=1)
    for i in range(nrow):
        for j in range(ncol):
            if bip_matrix[i, j] == 1:
                G.add_edge(i, nrow + j)
    return G


@pytest.fixture
def und_matrix() -> np.ndarray:
    """5×5 symmetric binary adjacency matrix (no self-loops)."""
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
def und_sparse(und_matrix):
    return sp.csr_matrix(und_matrix)


@pytest.fixture
def und_igraph(und_matrix):
    igraph = pytest.importorskip("igraph")
    n = und_matrix.shape[0]
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if und_matrix[i, j] == 1
    ]
    return igraph.Graph(n=n, edges=edges, directed=False)


@pytest.fixture
def und_nx(und_matrix):
    nx = pytest.importorskip("networkx")
    n = und_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if und_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G


# ---- Bipartite sparse (scipy) ----


def test_bip_scipy_output_type(bip_sparse):
    result = rewire_bipartite(bip_sparse, verbose=False, seed=1)
    assert sp.issparse(result)
    assert result.format == bip_sparse.format


def test_bip_scipy_row_sums_preserved(bip_sparse, bip_matrix):
    result = rewire_bipartite(bip_sparse, verbose=False, seed=1)
    np.testing.assert_array_equal(
        np.asarray(result.sum(axis=1)).ravel(),
        bip_matrix.sum(axis=1),
    )


def test_bip_scipy_col_sums_preserved(bip_sparse, bip_matrix):
    result = rewire_bipartite(bip_sparse, verbose=False, seed=1)
    np.testing.assert_array_equal(
        np.asarray(result.sum(axis=0)).ravel(),
        bip_matrix.sum(axis=0),
    )


def test_bip_scipy_deterministic(bip_sparse):
    r1 = rewire_bipartite(bip_sparse, verbose=False, seed=42)
    r2 = rewire_bipartite(bip_sparse, verbose=False, seed=42)
    np.testing.assert_array_equal(r1.toarray(), r2.toarray())


# ---- Bipartite igraph ----


def test_bip_igraph_output_type(bip_igraph):
    igraph = pytest.importorskip("igraph")
    result = rewire_bipartite(bip_igraph, verbose=False, seed=1)
    assert isinstance(result, igraph.Graph)


def test_bip_igraph_degree_preserved(bip_igraph):
    result = rewire_bipartite(bip_igraph, verbose=False, seed=1)
    orig_deg = sorted(bip_igraph.degree())
    new_deg = sorted(result.degree())
    assert orig_deg == new_deg


def test_bip_igraph_edge_count_preserved(bip_igraph):
    result = rewire_bipartite(bip_igraph, verbose=False, seed=1)
    assert result.ecount() == bip_igraph.ecount()


# ---- Bipartite networkx ----


def test_bip_nx_output_type(bip_nx):
    nx = pytest.importorskip("networkx")
    result = rewire_bipartite(bip_nx, verbose=False, seed=1)
    assert isinstance(result, nx.Graph)


def test_bip_nx_degree_preserved(bip_nx):
    result = rewire_bipartite(bip_nx, verbose=False, seed=1)
    orig_deg = sorted(d for _, d in bip_nx.degree())
    new_deg = sorted(d for _, d in result.degree())
    assert orig_deg == new_deg


def test_bip_nx_bipartite_attribute_preserved(bip_nx):
    pytest.importorskip("networkx")
    result = rewire_bipartite(bip_nx, verbose=False, seed=1)
    orig_parts = {
        n: d["bipartite"] for n, d in bip_nx.nodes(data=True)
    }
    new_parts = {
        n: d["bipartite"] for n, d in result.nodes(data=True)
    }
    assert orig_parts == new_parts


# ---- Undirected sparse (scipy) ----


def test_und_scipy_output_type(und_sparse):
    result = rewire_undirected(und_sparse, verbose=False, seed=1)
    assert sp.issparse(result)
    assert result.format == und_sparse.format


def test_und_scipy_degree_preserved(und_sparse):
    result = rewire_undirected(und_sparse, verbose=False, seed=1)
    orig_deg = sorted(np.asarray(und_sparse.sum(axis=1)).ravel().tolist())
    new_deg = sorted(np.asarray(result.sum(axis=1)).ravel().tolist())
    assert orig_deg == new_deg


def test_und_scipy_symmetric(und_sparse):
    result = rewire_undirected(und_sparse, verbose=False, seed=1)
    diff = (result - result.T).toarray()
    np.testing.assert_array_equal(diff, 0)


def test_und_scipy_deterministic(und_sparse):
    r1 = rewire_undirected(und_sparse, verbose=False, seed=99)
    r2 = rewire_undirected(und_sparse, verbose=False, seed=99)
    np.testing.assert_array_equal(r1.toarray(), r2.toarray())


# ---- Undirected igraph ----


def test_und_igraph_output_type(und_igraph):
    igraph = pytest.importorskip("igraph")
    result = rewire_undirected(und_igraph, verbose=False, seed=1)
    assert isinstance(result, igraph.Graph)


def test_und_igraph_degree_preserved(und_igraph):
    result = rewire_undirected(und_igraph, verbose=False, seed=1)
    orig_deg = sorted(und_igraph.degree())
    new_deg = sorted(result.degree())
    assert orig_deg == new_deg


def test_und_igraph_no_self_loops(und_igraph):
    result = rewire_undirected(und_igraph, verbose=False, seed=1)
    assert result.is_simple()


# ---- Undirected networkx ----


def test_und_nx_output_type(und_nx):
    nx = pytest.importorskip("networkx")
    result = rewire_undirected(und_nx, verbose=False, seed=1)
    assert isinstance(result, nx.Graph)


def test_und_nx_degree_preserved(und_nx):
    result = rewire_undirected(und_nx, verbose=False, seed=1)
    orig_deg = sorted(d for _, d in und_nx.degree())
    new_deg = sorted(d for _, d in result.degree())
    assert orig_deg == new_deg


def test_und_nx_node_set_preserved(und_nx):
    result = rewire_undirected(und_nx, verbose=False, seed=1)
    assert set(result.nodes()) == set(und_nx.nodes())
