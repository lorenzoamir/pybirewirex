"""Pure-NumPy fallback implementations of the BiRewire switching algorithm.

Used automatically when the C extension is unavailable (_C_AVAILABLE = False).
Results are statistically equivalent to the C backend but not bit-identical
(different PRNG: NumPy PCG64 vs C xorshift64).
"""

from __future__ import annotations

import sys

import numpy as np


def _progress(n: int, N: int) -> None:
    if N < 100 or n % (N // 100) != 0:
        return
    pct = n * 100 // N
    bar = "=" * (pct // 2) + " " * (50 - pct // 2)
    sys.stderr.write(f"\r{pct:3d}% [{bar}]")


# ---- Dense bipartite ---------------------------------------------------


def rewire_bipartite(
    m: np.ndarray, N: int, seed: int, verbose: bool = False
) -> np.ndarray:
    """Bipartite dense switching algorithm (NumPy implementation)."""
    rng = np.random.default_rng(seed)
    m = m.astype(np.int16, copy=True)
    rows, cols = np.where(m == 1)
    from_arr = rows.tolist()
    to_arr = cols.tolist()
    e = len(from_arr)
    if e < 2:
        return m

    for n in range(N):
        if verbose:
            _progress(n, N)
        r1 = int(rng.integers(e))
        r2 = int(rng.integers(e - 1))
        if r2 >= r1:
            r2 += 1
        a, b = from_arr[r1], to_arr[r1]
        c, d = from_arr[r2], to_arr[r2]
        if a != c and d != b and m[a, d] == 0 and m[c, b] == 0:
            to_arr[r1] = d
            to_arr[r2] = b
            m[a, d] = m[c, b] = 1
            m[a, b] = m[c, d] = 0

    if verbose:
        sys.stderr.write("\nDONE\n")
    return m


def analysis_bipartite(
    m: np.ndarray,
    N: int,
    ns: int,
    step: int,
    seed: int,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """Bipartite convergence analysis (NumPy). Returns (scores_1d, n_written)."""
    from pybirewirex.similarity import jaccard  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    orig = m.astype(np.int16, copy=True)
    working = orig.copy()
    rows, cols = np.where(orig == 1)
    from_arr = rows.tolist()
    to_arr = cols.tolist()
    e = len(from_arr)

    scores = np.zeros(ns, dtype=np.float64)
    scores[0] = 1.0
    index = 1
    if e < 2:
        return scores, index

    for n in range(N):
        if verbose:
            _progress(n, N)
        r1 = int(rng.integers(e))
        r2 = int(rng.integers(e - 1))
        if r2 >= r1:
            r2 += 1
        a, b = from_arr[r1], to_arr[r1]
        c, d = from_arr[r2], to_arr[r2]
        if a != c and d != b and working[a, d] == 0 and working[c, b] == 0:
            to_arr[r1] = d
            to_arr[r2] = b
            working[a, d] = working[c, b] = 1
            working[a, b] = working[c, d] = 0
        if n % step == 0 and index < ns:
            scores[index] = jaccard(working, orig)
            index += 1

    if verbose:
        sys.stderr.write("\nDONE\n")
    return scores, index


# ---- Dense undirected --------------------------------------------------


def rewire_undirected(
    m: np.ndarray, N: int, seed: int, verbose: bool = False
) -> np.ndarray:
    """Undirected dense switching algorithm (NumPy implementation)."""
    rng = np.random.default_rng(seed)
    m = m.astype(np.int16, copy=True)
    # Lower triangle (from > to), matching C convention
    fi, ti = np.where(np.tril(m, k=-1) == 1)
    from_arr = fi.tolist()
    to_arr = ti.tolist()
    e = len(from_arr)
    if e < 2:
        return m

    for n in range(N):
        if verbose:
            _progress(n, N)
        r1 = int(rng.integers(e))
        r2 = int(rng.integers(e - 1))
        if r2 >= r1:
            r2 += 1
        a, b = from_arr[r1], to_arr[r1]
        c, d = from_arr[r2], to_arr[r2]
        if a == c or b == d or a == d or c == b:
            continue
        path1_ok = m[a, d] == 0 and m[c, b] == 0
        path2_ok = m[a, c] == 0 and m[d, b] == 0
        if not (path1_ok or path2_ok):
            continue
        if path1_ok and path2_ok:
            path1 = bool(rng.random() >= 0.5)
        else:
            path1 = path1_ok
        if path1:
            m[a, d] = m[d, a] = m[c, b] = m[b, c] = 1
            m[a, b] = m[b, a] = m[c, d] = m[d, c] = 0
            to_arr[r1] = d
            to_arr[r2] = b
        else:
            m[a, c] = m[c, a] = m[d, b] = m[b, d] = 1
            m[a, b] = m[b, a] = m[c, d] = m[d, c] = 0
            to_arr[r1] = c
            from_arr[r2] = b
            to_arr[r2] = d

    if verbose:
        sys.stderr.write("\nDONE\n")
    return m


def analysis_undirected(
    m: np.ndarray,
    N: int,
    ns: int,
    step: int,
    seed: int,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """Undirected convergence analysis (NumPy). Returns (scores_1d, n_written)."""
    from pybirewirex.similarity import jaccard  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    orig = m.astype(np.int16, copy=True)
    working = orig.copy()
    fi, ti = np.where(np.tril(orig, k=-1) == 1)
    from_arr = fi.tolist()
    to_arr = ti.tolist()
    e = len(from_arr)

    scores = np.zeros(ns, dtype=np.float64)
    scores[0] = 1.0
    index = 1
    if e < 2:
        return scores, index

    for n in range(N):
        if verbose:
            _progress(n, N)
        r1 = int(rng.integers(e))
        r2 = int(rng.integers(e - 1))
        if r2 >= r1:
            r2 += 1
        a, b = from_arr[r1], to_arr[r1]
        c, d = from_arr[r2], to_arr[r2]
        if not (a == c or b == d or a == d or c == b):
            path1_ok = working[a, d] == 0 and working[c, b] == 0
            path2_ok = working[a, c] == 0 and working[d, b] == 0
            if path1_ok or path2_ok:
                if path1_ok and path2_ok:
                    path1 = bool(rng.random() >= 0.5)
                else:
                    path1 = path1_ok
                if path1:
                    working[a, d] = working[d, a] = working[c, b] = working[b, c] = 1
                    working[a, b] = working[b, a] = working[c, d] = working[d, c] = 0
                    to_arr[r1] = d
                    to_arr[r2] = b
                else:
                    working[a, c] = working[c, a] = working[d, b] = working[b, d] = 1
                    working[a, b] = working[b, a] = working[c, d] = working[d, c] = 0
                    to_arr[r1] = c
                    from_arr[r2] = b
                    to_arr[r2] = d
        if n % step == 0 and index < ns:
            scores[index] = jaccard(working, orig)
            index += 1

    if verbose:
        sys.stderr.write("\nDONE\n")
    return scores, index


# ---- Sparse bipartite --------------------------------------------------


def rewire_sparse_bipartite(
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    N: int,
    seed: int,
    verbose: bool = False,
) -> None:
    """Sparse bipartite switching algorithm (NumPy). Modifies from_arr/to_arr in-place."""
    rng = np.random.default_rng(seed)
    e = len(from_arr)
    if e < 2:
        return

    # Set of (from, to) for O(1) edge existence check
    edge_set: set[tuple[int, int]] = set(zip(from_arr.tolist(), to_arr.tolist()))

    for n in range(N):
        if verbose:
            _progress(n, N)
        r1 = int(rng.integers(e))
        r2 = int(rng.integers(e - 1))
        if r2 >= r1:
            r2 += 1
        a, b = int(from_arr[r1]), int(to_arr[r1])
        c, d = int(from_arr[r2]), int(to_arr[r2])
        if a != c and d != b and (a, d) not in edge_set and (c, b) not in edge_set:
            edge_set.discard((a, b))
            edge_set.discard((c, d))
            edge_set.add((a, d))
            edge_set.add((c, b))
            to_arr[r1] = d
            to_arr[r2] = b

    if verbose:
        sys.stderr.write("\nDONE\n")


# ---- Sparse undirected -------------------------------------------------


def rewire_sparse_undirected(
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    N: int,
    seed: int,
    verbose: bool = False,
) -> None:
    """Sparse undirected switching algorithm (NumPy). Modifies from_arr/to_arr in-place."""
    rng = np.random.default_rng(seed)
    e = len(from_arr)
    if e < 2:
        return

    # Symmetric adjacency set: both (a,b) and (b,a) for each edge
    adj_set: set[tuple[int, int]] = set()
    for i in range(e):
        a, b = int(from_arr[i]), int(to_arr[i])
        adj_set.add((a, b))
        adj_set.add((b, a))

    for n in range(N):
        if verbose:
            _progress(n, N)
        r1 = int(rng.integers(e))
        r2 = int(rng.integers(e - 1))
        if r2 >= r1:
            r2 += 1
        a, b = int(from_arr[r1]), int(to_arr[r1])
        c, d = int(from_arr[r2]), int(to_arr[r2])
        if a == c or b == d or a == d or c == b:
            continue
        # is_not(a,d): edge a-d absent; is_not(c,b): edge c-b absent
        # is_not(c,a): edge c-a absent; is_not(b,d): edge b-d absent
        ad = (a, d) not in adj_set
        cb = (c, b) not in adj_set
        ac = (a, c) not in adj_set
        bd = (b, d) not in adj_set
        if not ((ad and cb) or (ac and bd)):
            continue
        if ad and cb and ac and bd:
            path1 = bool(rng.random() >= 0.5)
        else:
            path1 = bool(ad and cb)
        if path1:
            adj_set.discard((a, b))
            adj_set.discard((b, a))
            adj_set.discard((c, d))
            adj_set.discard((d, c))
            adj_set.add((a, d))
            adj_set.add((d, a))
            adj_set.add((c, b))
            adj_set.add((b, c))
            to_arr[r1] = d
            to_arr[r2] = b
        else:
            adj_set.discard((a, b))
            adj_set.discard((b, a))
            adj_set.discard((c, d))
            adj_set.discard((d, c))
            adj_set.add((a, c))
            adj_set.add((c, a))
            adj_set.add((b, d))
            adj_set.add((d, b))
            to_arr[r1] = c
            from_arr[r2] = b
            to_arr[r2] = d

    if verbose:
        sys.stderr.write("\nDONE\n")
