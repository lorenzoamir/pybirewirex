# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-27

Initial release.

### Added

- `rewire_bipartite` — degree-preserving randomisation of bipartite networks
  (row and column sums preserved).
- `rewire_undirected` — degree-preserving randomisation of undirected graphs.
- `analysis_bipartite` / `analysis_undirected` — convergence analysis: record
  Jaccard similarity between the original network and its rewired snapshots
  across switching steps, returning mean trajectories and the analytical bound N.
- Analytical bound formula for both bipartite and undirected networks
  (`pybirewirex._bounds`), matching R's `exact=FALSE` default.
- Jaccard similarity helper (`pybirewirex.similarity.jaccard`).
- Input-type dispatch: all rewiring functions accept `numpy.ndarray`,
  `scipy.sparse` matrices, `igraph.Graph`, and `networkx.Graph`.
- C backend (`c_src/birewire_core.c`) based on the original BiRewire C code
  with R dependencies replaced by a standalone xorshift64 PRNG; compiled via
  CFFI.
- Pure-NumPy fallback (`pybirewirex._numpy_fallback`) activated automatically
  when the compiled extension is unavailable.
- 127 unit and regression tests against R reference outputs.

### Implementation notes

- The switching algorithm counts **total attempts** (not successful swaps),
  matching R's `exact=FALSE` / `MAXITER=0` default path.
- PRNG: R uses Mersenne Twister; this package uses xorshift64.
  Individual rewired matrices differ; the stationary distribution is identical.
- DSG (directed signed graph) rewiring is not yet implemented.
