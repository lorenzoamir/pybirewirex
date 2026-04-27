/*
 * Standalone header for BiRewire core algorithms.
 * Derived from BiRewire (Andrea Gobbi, 2013), GPL-3.
 * R runtime dependencies removed for standalone use.
 *
 * Dense functions accept flat column-major int16_t arrays:
 *   element (row i, col j) is at flat[j * nrow + i]
 *
 * MAXITER=0 → standard mode (count all iterations toward max_iter)
 * MAXITER>0 → extended mode (count only successful rewirings; abort if
 *             total attempts exceed MAXITER)
 *
 * Return values for rewire functions: 0 success, -1 MAXITER exceeded, -2 OOM.
 * Return values for analysis functions: number of scores written (>=1) on
 *   success, -1 MAXITER exceeded, -2 OOM.
 */
#ifndef BIREWIRE_CORE_H
#define BIREWIRE_CORE_H

#include <stddef.h>
#include <stdint.h>

int bw_rewire_bipartite(int16_t *flat, size_t ncol, size_t nrow,
                        size_t max_iter, int verbose, size_t MAXITER,
                        uint64_t seed);

int bw_analysis_bipartite(int16_t *flat, size_t ncol, size_t nrow,
                          double *scores, size_t step, size_t max_iter,
                          int verbose, size_t MAXITER, uint64_t seed);

int bw_rewire_undirected(int16_t *flat, size_t ncol, size_t nrow,
                         size_t max_iter, int verbose, size_t MAXITER,
                         uint64_t seed);

int bw_analysis_undirected(int16_t *flat, size_t ncol, size_t nrow,
                           double *scores, size_t step, size_t max_iter,
                           int verbose, size_t MAXITER, uint64_t seed);

/* Sparse bipartite: from/to are sorted COO (sorted by from[]).
   Edges are 0-indexed. */
int bw_rewire_sparse_bipartite(size_t *from, size_t *to,
                               size_t nc, size_t nr,
                               size_t max_iter, size_t e,
                               int verbose, size_t MAXITER,
                               uint64_t seed);

/* Sparse undirected: from/to are sorted COO, degree[i] = degree of node i. */
int bw_rewire_sparse(size_t *from, size_t *to, size_t *degree,
                     size_t nc, size_t nr,
                     size_t max_iter, size_t e,
                     int verbose, size_t MAXITER,
                     uint64_t seed);

#endif /* BIREWIRE_CORE_H */
