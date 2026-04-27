"""Dense undirected rewiring and convergence analysis."""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from pybirewirex._bounds import bound_undirected
from pybirewirex._core import _C_AVAILABLE, ffi, lib
from pybirewirex.bipartite import AnalysisResult, _make_seed, _n_steps


def _resolve_max_iter(
    max_iter: Union[int, str], e: int, t: int, accuracy: float, exact: bool
) -> int:
    if max_iter == "n":
        return bound_undirected(e, t, accuracy, exact)
    return int(max_iter)


def rewire_undirected(
    adjacency: Union[np.ndarray, Any],
    max_iter: Union[int, str] = "n",
    accuracy: float = 1e-5,
    exact: bool = False,
    verbose: bool = True,
    seed: int | None = None,
) -> Any:
    """Rewire an undirected network preserving degree sequence.

    Args:
        adjacency: 2-D binary symmetric ndarray, scipy sparse matrix,
            igraph.Graph, or networkx.Graph.
        max_iter: number of iterations, or "n" to auto-compute the bound.
        accuracy: convergence accuracy used when max_iter="n".
        exact: use exact bound formula when max_iter="n".
        verbose: print progress to stderr.
        seed: integer seed for reproducibility; None uses os.urandom.

    Returns:
        Rewired network in the same type as *adjacency*.
    """
    if not isinstance(adjacency, np.ndarray):
        from pybirewirex.sparse import is_sparse_or_graph, rewire_undirected_sparse  # noqa: PLC0415

        if is_sparse_or_graph(adjacency):
            return rewire_undirected_sparse(
                adjacency,
                max_iter=max_iter,
                accuracy=accuracy,
                exact=exact,
                verbose=verbose,
                seed=seed,
            )
        raise TypeError(f"Unsupported input type: {type(adjacency)!r}")

    m = np.asarray(adjacency, dtype=np.int16)
    n = m.shape[0]
    # edges = upper triangle sum
    e = int(np.sum(np.triu(m, k=1)))
    t = n * (n - 1) // 2

    N = _resolve_max_iter(max_iter, e, t, accuracy, exact)

    if not _C_AVAILABLE:
        import pybirewirex._numpy_fallback as _fb  # noqa: PLC0415

        return _fb.rewire_undirected(m, N, _make_seed(seed), verbose)

    flat = np.ascontiguousarray(m.T, dtype=np.int16).ravel()
    buf = ffi.from_buffer("int16_t[]", flat)

    seed_val = _make_seed(seed)
    ret = lib.bw_rewire_undirected(buf, n, n, N, int(verbose), 0, seed_val)
    if ret == -2:
        raise MemoryError("C backend out of memory")

    result = np.ascontiguousarray(flat.reshape((n, n)).T)
    out_dtype = np.asarray(adjacency).dtype
    return result.astype(out_dtype) if result.dtype != out_dtype else result


def analysis_undirected(
    adjacency: np.ndarray,
    step: int = 10,
    max_iter: Union[int, str] = "n",
    n_networks: int = 50,
    accuracy: float = 1e-5,
    exact: bool = False,
    verbose: bool = True,
    seed: int | None = None,
) -> AnalysisResult:
    """Run convergence analysis for dense undirected rewiring.

    Runs the Switching Algorithm for *n_networks* independent rewirings,
    recording Jaccard similarity between the original and rewired adjacency
    matrix every *step* iterations.

    Args:
        adjacency: 2-D binary symmetric ndarray of shape (n, n), no self-loops.
        step: record Jaccard every *step* iterations.
        max_iter: total iterations per network, or "n" to use the bound.
        n_networks: number of independent rewirings.
        accuracy: convergence accuracy used when max_iter="n".
        exact: use exact bound formula when max_iter="n".
        verbose: print progress to stderr.
        seed: integer seed; each network gets seed+i for reproducibility.

    Returns:
        AnalysisResult with N (recommended iterations), scores
        (shape n_networks × n_steps), and step.
    """
    m = np.asarray(adjacency, dtype=np.int16)
    n = m.shape[0]
    e = int(np.sum(np.triu(m, k=1)))
    t = n * (n - 1) // 2

    # N_bound is always the analytical bound, stored in the result for plotting.
    # n_run is what we actually iterate — may be larger when max_iter is explicit.
    N_bound = bound_undirected(e, t, accuracy, exact)
    n_run   = _resolve_max_iter(max_iter, e, t, accuracy, exact)
    ns = _n_steps(n_run, step)
    base_seed = _make_seed(seed)

    if not _C_AVAILABLE:
        import pybirewirex._numpy_fallback as _fb  # noqa: PLC0415

        all_scores = np.zeros((n_networks, ns), dtype=np.float64)
        for i in range(n_networks):
            net_seed = (base_seed + i) & 0xFFFFFFFFFFFFFFFF
            scores_1d, n_written = _fb.analysis_undirected(m, n_run, ns, step, net_seed, verbose)
            all_scores[i, :n_written] = scores_1d[:n_written]
        return AnalysisResult(N=N_bound, scores=all_scores, step=step)

    all_scores = np.zeros((n_networks, ns), dtype=np.float64)

    for i in range(n_networks):
        flat = np.ascontiguousarray(m.T, dtype=np.int16).ravel()
        buf = ffi.from_buffer("int16_t[]", flat)
        scores_buf = ffi.new(f"double[{ns}]")

        net_seed = (base_seed + i) & 0xFFFFFFFFFFFFFFFF
        ret = lib.bw_analysis_undirected(
            buf, n, n, scores_buf, step, n_run, int(verbose), 0, net_seed
        )
        if ret == -2:
            raise MemoryError("C backend out of memory")

        n_written = ret if ret > 0 else 0
        for k in range(n_written):
            all_scores[i, k] = scores_buf[k]

    return AnalysisResult(N=N_bound, scores=all_scores, step=step)
