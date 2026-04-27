"""Sparse and graph input dispatch for rewiring functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from pybirewirex._bounds import bound_bipartite, bound_undirected
from pybirewirex._core import _C_AVAILABLE, ffi, lib
from pybirewirex.bipartite import _make_seed


def _is_scipy_sparse(x: Any) -> bool:
    try:
        import scipy.sparse as sp  # noqa: PLC0415

        return sp.issparse(x)
    except ImportError:
        return False


def _is_igraph(x: Any) -> bool:
    try:
        import igraph  # noqa: PLC0415

        return isinstance(x, igraph.Graph)
    except ImportError:
        return False


def _is_networkx(x: Any) -> bool:
    try:
        import networkx as nx  # noqa: PLC0415

        return isinstance(x, nx.Graph)
    except ImportError:
        return False


def is_sparse_or_graph(x: Any) -> bool:
    """Return True if x is a scipy sparse matrix, igraph Graph, or networkx Graph."""
    return _is_scipy_sparse(x) or _is_igraph(x) or _is_networkx(x)


# ---- Bipartite COO extraction ----


def _bipartite_to_coo(
    graph: Any,
) -> tuple[np.ndarray, np.ndarray, int, int, dict]:
    """Convert bipartite graph to sorted COO edge list.

    Returns (from_arr, to_arr, nrow, ncol, meta).
    from_arr is sorted ascending (required by C sparse bipartite).
    Both arrays are uint64, writable, C-contiguous.
    """
    if _is_scipy_sparse(graph):
        fmt = graph.format
        coo = graph.asformat("coo")
        row = np.array(coo.row, dtype=np.uint64)
        col = np.array(coo.col, dtype=np.uint64)
        nrow, ncol = coo.shape
        order = np.argsort(row, kind="stable")
        return (
            np.ascontiguousarray(row[order]),
            np.ascontiguousarray(col[order]),
            nrow,
            ncol,
            {"input_type": "scipy", "fmt": fmt, "shape": (nrow, ncol)},
        )

    if _is_networkx(graph):
        nodes0 = sorted(n for n, d in graph.nodes(data=True) if d.get("bipartite") == 0)
        nodes1 = sorted(n for n, d in graph.nodes(data=True) if d.get("bipartite") == 1)
        map0 = {n: i for i, n in enumerate(nodes0)}
        map1 = {n: i for i, n in enumerate(nodes1)}
        rows: list[int] = []
        cols: list[int] = []
        for u, v in graph.edges():
            if u in map0 and v in map1:
                rows.append(map0[u])
                cols.append(map1[v])
            elif v in map0 and u in map1:
                rows.append(map0[v])
                cols.append(map1[u])
        row = np.array(rows, dtype=np.uint64)
        col = np.array(cols, dtype=np.uint64)
        order = np.argsort(row, kind="stable")
        return (
            np.ascontiguousarray(row[order]),
            np.ascontiguousarray(col[order]),
            len(nodes0),
            len(nodes1),
            {
                "input_type": "networkx",
                "nodes0": nodes0,
                "nodes1": nodes1,
                "directed": graph.is_directed(),
            },
        )

    if _is_igraph(graph):
        types = [bool(t) for t in graph.vs["type"]]
        nodes0 = [v.index for v in graph.vs if not types[v.index]]
        nodes1 = [v.index for v in graph.vs if types[v.index]]
        map0 = {n: i for i, n in enumerate(nodes0)}
        map1 = {n: i for i, n in enumerate(nodes1)}
        rows2: list[int] = []
        cols2: list[int] = []
        for e in graph.es:
            s, t_v = e.source, e.target
            if s in map0 and t_v in map1:
                rows2.append(map0[s])
                cols2.append(map1[t_v])
            elif t_v in map0 and s in map1:
                rows2.append(map0[t_v])
                cols2.append(map1[s])
        row = np.array(rows2, dtype=np.uint64)
        col = np.array(cols2, dtype=np.uint64)
        order = np.argsort(row, kind="stable")
        return (
            np.ascontiguousarray(row[order]),
            np.ascontiguousarray(col[order]),
            len(nodes0),
            len(nodes1),
            {
                "input_type": "igraph",
                "nodes0": nodes0,
                "nodes1": nodes1,
                "directed": graph.is_directed(),
            },
        )

    raise TypeError(f"Unsupported graph type: {type(graph)!r}")


def _bipartite_from_coo(from_arr: np.ndarray, to_arr: np.ndarray, meta: dict) -> Any:
    """Reconstruct original bipartite type from rewired edge list."""
    if meta["input_type"] == "scipy":
        import scipy.sparse as sp  # noqa: PLC0415

        nrow, ncol = meta["shape"]
        e = len(from_arr)
        data = np.ones(e, dtype=np.int8)
        mat = sp.coo_matrix(
            (data, (from_arr.astype(np.intp), to_arr.astype(np.intp))),
            shape=(nrow, ncol),
        )
        return mat.asformat(meta["fmt"])

    if meta["input_type"] == "networkx":
        import networkx as nx  # noqa: PLC0415

        nodes0, nodes1 = meta["nodes0"], meta["nodes1"]
        G = nx.DiGraph() if meta["directed"] else nx.Graph()
        G.add_nodes_from(nodes0, bipartite=0)
        G.add_nodes_from(nodes1, bipartite=1)
        for r, c in zip(from_arr.tolist(), to_arr.tolist()):
            G.add_edge(nodes0[r], nodes1[c])
        return G

    if meta["input_type"] == "igraph":
        import igraph  # noqa: PLC0415

        nodes0, nodes1 = meta["nodes0"], meta["nodes1"]
        n = len(nodes0) + len(nodes1)
        edges = [(int(nodes0[r]), int(nodes1[c])) for r, c in zip(from_arr, to_arr)]
        G = igraph.Graph(n=n, edges=edges, directed=meta["directed"])
        type_arr: list[bool | None] = [None] * n
        for v in nodes0:
            type_arr[v] = False
        for v in nodes1:
            type_arr[v] = True
        G.vs["type"] = type_arr
        return G

    raise ValueError(f"Unknown input type: {meta['input_type']!r}")


# ---- Undirected COO extraction ----


def _undirected_to_coo(
    graph: Any,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, dict]:
    """Convert undirected graph to COO edge list (each edge stored once).

    Returns (from_arr, to_arr, n_nodes, degree_arr, meta).
    All arrays are uint64, writable, C-contiguous.
    """
    if _is_scipy_sparse(graph):
        import scipy.sparse as sp  # noqa: PLC0415

        fmt = graph.format
        n = graph.shape[0]
        upper = sp.triu(graph, k=1).asformat("coo")
        row = np.array(upper.row, dtype=np.uint64)
        col = np.array(upper.col, dtype=np.uint64)
        csr = graph.asformat("csr")
        deg = np.asarray(csr.sum(axis=1), dtype=np.float64).ravel()
        deg_u = np.round(deg).astype(np.uint64)
        return (
            np.ascontiguousarray(row),
            np.ascontiguousarray(col),
            n,
            np.ascontiguousarray(deg_u),
            {"input_type": "scipy", "fmt": fmt, "n": n},
        )

    if _is_networkx(graph):
        nodes = sorted(graph.nodes())
        map_v = {n: i for i, n in enumerate(nodes)}
        rows_u: list[int] = []
        cols_u: list[int] = []
        for u, v in graph.edges():
            a, b = map_v[u], map_v[v]
            rows_u.append(min(a, b))
            cols_u.append(max(a, b))
        row = np.array(rows_u, dtype=np.uint64)
        col = np.array(cols_u, dtype=np.uint64)
        deg = np.array([graph.degree(n) for n in nodes], dtype=np.uint64)
        return (
            np.ascontiguousarray(row),
            np.ascontiguousarray(col),
            len(nodes),
            np.ascontiguousarray(deg),
            {
                "input_type": "networkx",
                "nodes": nodes,
                "directed": graph.is_directed(),
            },
        )

    if _is_igraph(graph):
        n = graph.vcount()
        edges = graph.get_edgelist()
        rows_i = [min(a, b) for a, b in edges]
        cols_i = [max(a, b) for a, b in edges]
        row = np.array(rows_i, dtype=np.uint64)
        col = np.array(cols_i, dtype=np.uint64)
        deg = np.array(graph.degree(), dtype=np.uint64)
        return (
            np.ascontiguousarray(row),
            np.ascontiguousarray(col),
            n,
            np.ascontiguousarray(deg),
            {"input_type": "igraph", "n": n, "directed": graph.is_directed()},
        )

    raise TypeError(f"Unsupported graph type: {type(graph)!r}")


def _undirected_from_coo(
    from_arr: np.ndarray, to_arr: np.ndarray, n: int, meta: dict
) -> Any:
    """Reconstruct original undirected graph from rewired edge list."""
    if meta["input_type"] == "scipy":
        import scipy.sparse as sp  # noqa: PLC0415

        n_nodes = meta["n"]
        r = np.concatenate([from_arr, to_arr]).astype(np.intp)
        c = np.concatenate([to_arr, from_arr]).astype(np.intp)
        data = np.ones(len(r), dtype=np.int8)
        mat = sp.coo_matrix((data, (r, c)), shape=(n_nodes, n_nodes))
        return mat.asformat(meta["fmt"])

    if meta["input_type"] == "networkx":
        import networkx as nx  # noqa: PLC0415

        nodes = meta["nodes"]
        G = nx.DiGraph() if meta["directed"] else nx.Graph()
        G.add_nodes_from(nodes)
        for a, b in zip(from_arr.tolist(), to_arr.tolist()):
            G.add_edge(nodes[a], nodes[b])
        return G

    if meta["input_type"] == "igraph":
        import igraph  # noqa: PLC0415

        n_nodes = meta["n"]
        edges = list(zip(from_arr.tolist(), to_arr.tolist()))
        return igraph.Graph(n=n_nodes, edges=edges, directed=meta["directed"])

    raise ValueError(f"Unknown input type: {meta['input_type']!r}")


# ---- Public rewiring functions ----


def rewire_bipartite_sparse(
    graph: Any,
    max_iter: int | str = "n",
    accuracy: float = 1e-5,
    exact: bool = False,
    verbose: bool = True,
    seed: int | None = None,
) -> Any:
    """Rewire a bipartite sparse/graph input preserving row and column sums."""
    from_arr, to_arr, nrow, ncol, meta = _bipartite_to_coo(graph)
    e = len(from_arr)
    t = nrow * ncol

    N = bound_bipartite(e, t, accuracy, exact) if max_iter == "n" else int(max_iter)
    seed_val = _make_seed(seed)

    if not _C_AVAILABLE:
        import pybirewirex._numpy_fallback as _fb  # noqa: PLC0415

        _fb.rewire_sparse_bipartite(from_arr, to_arr, N, seed_val, verbose)
        return _bipartite_from_coo(from_arr, to_arr, meta)

    from_buf = ffi.from_buffer("size_t[]", from_arr)
    to_buf = ffi.from_buffer("size_t[]", to_arr)

    ret = lib.bw_rewire_sparse_bipartite(
        from_buf, to_buf, ncol, nrow, N, e, int(verbose), 0, seed_val
    )
    if ret == -2:
        raise MemoryError("C backend out of memory")

    return _bipartite_from_coo(from_arr, to_arr, meta)


def rewire_undirected_sparse(
    graph: Any,
    max_iter: int | str = "n",
    accuracy: float = 1e-5,
    exact: bool = False,
    verbose: bool = True,
    seed: int | None = None,
) -> Any:
    """Rewire an undirected sparse/graph input preserving degree sequence."""
    from_arr, to_arr, n, deg, meta = _undirected_to_coo(graph)
    e = len(from_arr)
    t = n * (n - 1) // 2

    N = bound_undirected(e, t, accuracy, exact) if max_iter == "n" else int(max_iter)
    seed_val = _make_seed(seed)

    if not _C_AVAILABLE:
        import pybirewirex._numpy_fallback as _fb  # noqa: PLC0415

        _fb.rewire_sparse_undirected(from_arr, to_arr, N, seed_val, verbose)
        return _undirected_from_coo(from_arr, to_arr, n, meta)

    from_buf = ffi.from_buffer("size_t[]", from_arr)
    to_buf = ffi.from_buffer("size_t[]", to_arr)
    deg_buf = ffi.from_buffer("size_t[]", deg)

    ret = lib.bw_rewire_sparse(
        from_buf, to_buf, deg_buf, n, n, N, e, int(verbose), 0, seed_val
    )
    if ret == -2:
        raise MemoryError("C backend out of memory")

    return _undirected_from_coo(from_arr, to_arr, n, meta)
