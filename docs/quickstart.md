# Quick start

## Bipartite rewiring

```python
import numpy as np
import pybirewirex as pbr

# Binary incidence matrix (rows = genes, cols = samples, for example)
rng    = np.random.default_rng(0)
matrix = (rng.random((100, 40)) < 0.2).astype(np.int16)

# Rewire — row and column sums are preserved
rewired = pbr.rewire_bipartite(matrix, seed=42)

assert np.array_equal(matrix.sum(axis=1), rewired.sum(axis=1))
assert np.array_equal(matrix.sum(axis=0), rewired.sum(axis=0))
```

## Undirected rewiring

```python
# Symmetric adjacency matrix (no self-loops)
adj     = (rng.random((50, 50)) < 0.05).astype(np.int16)
adj     = np.triu(adj, 1); adj = adj + adj.T   # symmetrise

rewired = pbr.rewire_undirected(adj, seed=42)

assert np.array_equal(adj.sum(axis=0), rewired.sum(axis=0))  # degrees
assert np.array_equal(rewired, rewired.T)                      # symmetric
```

## Convergence analysis

```python
result = pbr.analysis_bipartite(
    matrix,
    n_networks = 10,    # independent runs
    step       = 1000,  # record Jaccard every 1 000 steps
    seed       = 0,
)

print(f"Bound N = {result.N}")
print(f"Stationary Jaccard ≈ {result.scores[:, -1].mean():.3f}")
```

`result.scores` has shape `(n_networks, n_steps)`. The first column is always
1.0 (identity); the last column is at or past the bound N.

## Graph inputs

```python
import igraph as ig

g       = ig.Graph.Bipartite([0]*100 + [1]*40, [(i, j) for i in range(100) for j in range(40) if rng.random() < 0.2])
g_rw    = pbr.rewire_bipartite(g, seed=42)   # returns igraph.Graph

import networkx as nx
G       = nx.erdos_renyi_graph(200, 0.05, seed=0)
G_rw    = pbr.rewire_undirected(G, seed=42)  # returns networkx.Graph
```

## Reproducibility

Pass an integer `seed` to any function for deterministic results:

```python
r1 = pbr.rewire_bipartite(matrix, seed=7)
r2 = pbr.rewire_bipartite(matrix, seed=7)
assert np.array_equal(r1, r2)
```

`seed=None` (default) draws from `os.urandom`.
