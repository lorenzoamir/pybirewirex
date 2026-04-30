"""Dense bipartite rewiring and convergence analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import numpy as np

from pybirewirex._bounds import bound_bipartite
from pybirewirex._core import _C_AVAILABLE, ffi, lib


@dataclass
class AnalysisResult:
    N: int
    scores: np.ndarray  # shape: (n_networks, n_steps)
    step: int


def _resolve_max_iter(
    max_iter: int | str, e: int, t: int, accuracy: float, exact: bool
) -> int:
    if max_iter == "n":
        return bound_bipartite(e, t, accuracy, exact)
    return int(max_iter)


def _make_seed(seed: int | None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little")
    return seed & 0xFFFFFFFFFFFFFFFF


def _n_steps(N: int, step: int) -> int:
    # C writes scores[0]=1.0 plus one entry per step interval in [0, N).
    # Entries at n=0, step, 2*step, ... where n < N:
    #   count = floor((N-1)/step)+1 for N>0, else 0
    # Total = 1 + count.
    # Python floor division: (N-1)//step+2 works for N>=0 (N=0 → -1//s+2=1).
    return (N - 1) // step + 2 if N > 0 else 1


def rewire_bipartite(
    matrix: np.ndarray | Any,
    max_iter: int | str = "n",
    accuracy: float = 1e-5,
    exact: bool = False,
    verbose: bool = False,
    seed: int | None = None,
) -> Any:
    """Rewire a bipartite network preserving row and column sums.

    Args:
        matrix: 2-D binary ndarray, scipy sparse matrix, igraph.Graph, or
            networkx.Graph representing a bipartite network.
        max_iter: number of iterations, or "n" to auto-compute the bound.
        accuracy: convergence accuracy used when max_iter="n".
        exact: use exact bound formula when max_iter="n".
        verbose: print progress to stderr.
        seed: integer seed for reproducibility; None uses os.urandom.

    Returns:
        Rewired network in the same type as *matrix*.
    """
    if not isinstance(matrix, np.ndarray):
        from pybirewirex.sparse import (  # noqa: PLC0415
            is_sparse_or_graph,
            rewire_bipartite_sparse,
        )

        if is_sparse_or_graph(matrix):
            return rewire_bipartite_sparse(
                matrix,
                max_iter=max_iter,
                accuracy=accuracy,
                exact=exact,
                verbose=verbose,
                seed=seed,
            )
        raise TypeError(f"Unsupported input type: {type(matrix)!r}")

    m = np.asarray(matrix, dtype=np.int16)
    nrow, ncol = m.shape
    e = int(np.sum(m == 1))
    t = nrow * ncol

    N = _resolve_max_iter(max_iter, e, t, accuracy, exact)

    if not _C_AVAILABLE:
        import pybirewirex._numpy_fallback as _fb  # noqa: PLC0415

        return _fb.rewire_bipartite(m, N, _make_seed(seed), verbose)

    # Build column-major flat buffer: flat[j*nrow+i] = m[i,j].
    # Achieved by C-order ravel of the transposed matrix.
    flat = np.ascontiguousarray(m.T, dtype=np.int16).ravel()
    buf = ffi.from_buffer("int16_t[]", flat)

    seed_val = _make_seed(seed)
    ret = lib.bw_rewire_bipartite(buf, ncol, nrow, N, int(verbose), 0, seed_val)
    if ret == -2:
        raise MemoryError("C backend out of memory")

    # Reconstruct (nrow, ncol) C-order array from column-major flat buffer.
    result = np.ascontiguousarray(flat.reshape((ncol, nrow)).T)
    out_dtype = np.asarray(matrix).dtype
    return result.astype(out_dtype) if result.dtype != out_dtype else result


def analysis_bipartite(
    matrix: np.ndarray,
    step: int = 10,
    max_iter: int | str = "n",
    n_networks: int = 50,
    accuracy: float = 1e-5,
    exact: bool = False,
    verbose: bool = False,
    seed: int | None = None,
) -> AnalysisResult:
    """Run convergence analysis for dense bipartite rewiring.

    Runs the Switching Algorithm for *n_networks* independent rewirings,
    recording Jaccard similarity between the original and rewired matrix
    every *step* iterations.

    Args:
        matrix: 2-D binary ndarray of shape (nrow, ncol).
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
    m = np.asarray(matrix, dtype=np.int16)
    nrow, ncol = m.shape
    e = int(np.sum(m == 1))
    t = nrow * ncol

    # N_bound is always the analytical bound, stored in the result for plotting.
    # n_run is what we actually iterate — may be larger when max_iter is explicit.
    N_bound = bound_bipartite(e, t, accuracy, exact)
    n_run = _resolve_max_iter(max_iter, e, t, accuracy, exact)
    ns = _n_steps(n_run, step)
    base_seed = _make_seed(seed)

    if not _C_AVAILABLE:
        import pybirewirex._numpy_fallback as _fb  # noqa: PLC0415

        all_scores = np.zeros((n_networks, ns), dtype=np.float64)
        for i in range(n_networks):
            net_seed = (base_seed + i) & 0xFFFFFFFFFFFFFFFF
            scores_1d, n_written = _fb.analysis_bipartite(
                m, n_run, ns, step, net_seed, verbose
            )
            all_scores[i, :n_written] = scores_1d[:n_written]
        return AnalysisResult(N=N_bound, scores=all_scores, step=step)

    all_scores = np.zeros((n_networks, ns), dtype=np.float64)

    for i in range(n_networks):
        flat = np.ascontiguousarray(m.T, dtype=np.int16).ravel()
        buf = ffi.from_buffer("int16_t[]", flat)
        scores_buf = ffi.new(f"double[{ns}]")

        net_seed = (base_seed + i) & 0xFFFFFFFFFFFFFFFF
        ret = lib.bw_analysis_bipartite(
            buf, ncol, nrow, scores_buf, step, n_run, int(verbose), 0, net_seed
        )
        if ret == -2:
            raise MemoryError("C backend out of memory")

        n_written = ret if ret > 0 else 0
        for k in range(n_written):
            all_scores[i, k] = scores_buf[k]

    return AnalysisResult(N=N_bound, scores=all_scores, step=step)
