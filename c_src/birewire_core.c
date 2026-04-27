/*
 * Standalone C implementation of BiRewire network rewiring algorithms.
 * Derived from BiRewire.c (Andrea Gobbi <gobbi.andrea@mail.com>, 2013).
 * License: GPL-3.
 *
 * Changes from the original:
 *   - R headers and runtime removed (R_alloc → malloc/free,
 *     unif_rand → xorshift64, GetRNGstate/PutRNGstate → no-ops,
 *     Rprintf → fprintf(stderr), warning() → no-op)
 *   - All public functions take a uint64_t seed for reproducibility.
 *   - Dense functions take/return flat column-major int16_t arrays.
 *   - Analysis functions take a pre-allocated double* scores buffer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "birewire_core.h"

/* ---- xorshift64 PRNG -------------------------------------------- */

static inline uint64_t xorshift64_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

/* Returns a double in [0, 1) with 53 bits of precision. */
static inline double xrand(uint64_t *s) {
    return (double)(xorshift64_next(s) >> 11) * (1.0 / (double)(1ULL << 53));
}

static inline uint64_t make_seed(uint64_t seed) {
    /* xorshift64 state must never be 0 */
    return seed ? seed : 0xDEADBEEFCAFEBABEULL;
}

/* ---- Progress bar ----------------------------------------------- */

static void load_bar(size_t x, size_t n) {
    if (n < 100) return;
    if (x % (n / 100) != 0) return;
    int c = (int)((x * 50) / n);
    fprintf(stderr, "%3d%% [", (int)(x * 100 / n));
    for (int i = 0; i < c; i++) fputc('=', stderr);
    for (int i = c; i < 50; i++) fputc(' ', stderr);
    fprintf(stderr, "]\r");
}

/* ---- Jaccard similarity ------------------------------------------ */

static double jaccard_bipartite(short **a, short **b,
                                size_t nrow, size_t ncol, size_t e) {
    size_t num = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            if (a[i][j] == 1 && b[i][j] == 1) num++;
    return (double)num / (2.0 * e - (double)num);
}

static double jaccard_undirected(short **a, short **b,
                                 size_t nrow, size_t ncol, size_t e) {
    (void)ncol;
    size_t num = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < i; j++)
            if (a[i][j] == 1 && b[i][j] == 1) num++;
    return (double)num / (2.0 * e - (double)num);
}

/* ---- Matrix helpers --------------------------------------------- */

static short **flat_to_matrix(const int16_t *flat, size_t nrow, size_t ncol) {
    short **m = malloc(nrow * sizeof(short *));
    if (!m) return NULL;
    for (size_t i = 0; i < nrow; i++) {
        m[i] = malloc(ncol * sizeof(short));
        if (!m[i]) {
            for (size_t k = 0; k < i; k++) free(m[k]);
            free(m);
            return NULL;
        }
        /* flat is column-major: element (i,j) at flat[j*nrow+i] */
        for (size_t j = 0; j < ncol; j++)
            m[i][j] = (short)flat[j * nrow + i];
    }
    return m;
}

static void matrix_to_flat(short **m, int16_t *flat, size_t nrow, size_t ncol) {
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            flat[j * nrow + i] = (int16_t)m[i][j];
}

static short **copy_matrix(short **src, size_t nrow, size_t ncol) {
    short **m = malloc(nrow * sizeof(short *));
    if (!m) return NULL;
    for (size_t i = 0; i < nrow; i++) {
        m[i] = malloc(ncol * sizeof(short));
        if (!m[i]) {
            for (size_t k = 0; k < i; k++) free(m[k]);
            free(m);
            return NULL;
        }
        memcpy(m[i], src[i], ncol * sizeof(short));
    }
    return m;
}

static void free_matrix(short **m, size_t nrow) {
    if (!m) return;
    for (size_t i = 0; i < nrow; i++) free(m[i]);
    free(m);
}

/* ---- Sparse check helper (for bipartite sparse) ----------------- */

/* Returns 1 if from[rand1]→to[rand2] and from[rand2]→to[rand1] edges
   do not already exist.  pos[edge_idx] = row-group index;
   index[g]..index[g+1] is the slice of 'to' belonging to group g. */
static int check_sparse(const size_t *pos, const size_t *to,
                        const size_t *index,
                        size_t rand1, size_t b,
                        size_t rand2, size_t d) {
    for (size_t i = index[pos[rand1]]; i < index[pos[rand1] + 1]; i++)
        if (to[i] == d) return 0;
    for (size_t i = index[pos[rand2]]; i < index[pos[rand2] + 1]; i++)
        if (to[i] == b) return 0;
    return 1;
}

/* ---- Sparse undirected helpers ---------------------------------- */

static int is_not(size_t a, size_t d, const size_t *degree, short **adj) {
    /* Returns 1 if edge a-d is ABSENT, 0 if it exists. */
    for (size_t i = 0; i < degree[a]; i++)
        if ((size_t)adj[a][i] == d) return 0;
    return 1;
}

/* swap edge a-b to a-d, c-d to c-b */
static void sub1(size_t a, size_t b, size_t c, size_t d,
                 const size_t *degree, short **adj) {
    for (size_t i = 0; i < degree[a]; i++) if ((size_t)adj[a][i] == b) { adj[a][i] = (short)d; break; }
    for (size_t i = 0; i < degree[b]; i++) if ((size_t)adj[b][i] == a) { adj[b][i] = (short)c; break; }
    for (size_t i = 0; i < degree[c]; i++) if ((size_t)adj[c][i] == d) { adj[c][i] = (short)b; break; }
    for (size_t i = 0; i < degree[d]; i++) if ((size_t)adj[d][i] == c) { adj[d][i] = (short)a; break; }
}

/* swap edge a-b to a-c, d-b to d... (alternative rewiring) */
static void sub2(size_t a, size_t b, size_t c, size_t d,
                 const size_t *degree, short **adj) {
    for (size_t i = 0; i < degree[a]; i++) if ((size_t)adj[a][i] == b) { adj[a][i] = (short)c; break; }
    for (size_t i = 0; i < degree[b]; i++) if ((size_t)adj[b][i] == a) { adj[b][i] = (short)d; break; }
    for (size_t i = 0; i < degree[c]; i++) if ((size_t)adj[c][i] == d) { adj[c][i] = (short)a; break; }
    for (size_t i = 0; i < degree[d]; i++) if ((size_t)adj[d][i] == c) { adj[d][i] = (short)b; break; }
}

/* ================================================================
   PUBLIC API — DENSE BIPARTITE
   ================================================================ */

int bw_rewire_bipartite(int16_t *flat, size_t ncol, size_t nrow,
                        size_t max_iter, int verbose, size_t MAXITER,
                        uint64_t seed) {
    short **matrix = flat_to_matrix(flat, nrow, ncol);
    if (!matrix) return -2;

    size_t e = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            if (matrix[i][j] == 1) e++;

    size_t *from = malloc(e * sizeof(size_t));
    size_t *to   = malloc(e * sizeof(size_t));
    if (!from || !to) {
        free(from); free(to); free_matrix(matrix, nrow);
        return -2;
    }

    size_t kk = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            if (matrix[i][j] == 1) { from[kk] = i; to[kk] = j; kk++; }

    uint64_t state = make_seed(seed);
    int result = 0;

    if (MAXITER == 0) {
        /* Standard: every iteration counts toward max_iter */
        for (size_t n = 0; n < max_iter; n++) {
            if (verbose) load_bar(n, max_iter);
            size_t r1 = (size_t)(xrand(&state) * e);
            size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);
            size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];
            if (a != c && d != b && matrix[a][d] == 0 && matrix[c][b] == 0) {
                to[r1] = d; to[r2] = b;
                matrix[a][d] = matrix[c][b] = 1;
                matrix[a][b] = matrix[c][d] = 0;
            }
        }
    } else {
        /* Extended: count successful rewirings; total attempts capped at MAXITER */
        size_t n = 0, t = 0;
        while (n < max_iter) {
            if (verbose) load_bar(n, max_iter);
            size_t r1 = (size_t)(xrand(&state) * e);
            size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);
            size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];
            if (a != c && d != b && matrix[a][d] == 0 && matrix[c][b] == 0) {
                to[r1] = d; to[r2] = b;
                matrix[a][d] = matrix[c][b] = 1;
                matrix[a][b] = matrix[c][d] = 0;
                n++;
            }
            if (t++ > MAXITER) { result = -1; break; }
        }
    }

    if (verbose) fprintf(stderr, "\nDONE\n");
    matrix_to_flat(matrix, flat, nrow, ncol);
    free(from); free(to);
    free_matrix(matrix, nrow);
    return result;
}

int bw_analysis_bipartite(int16_t *flat, size_t ncol, size_t nrow,
                          double *scores, size_t step, size_t max_iter,
                          int verbose, size_t MAXITER, uint64_t seed) {
    /* orig = unchanged reference for Jaccard; working = gets rewired */
    short **orig = flat_to_matrix(flat, nrow, ncol);
    if (!orig) return -2;
    short **working = copy_matrix(orig, nrow, ncol);
    if (!working) { free_matrix(orig, nrow); return -2; }

    size_t e = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            if (orig[i][j] == 1) e++;

    size_t *from = malloc(e * sizeof(size_t));
    size_t *to   = malloc(e * sizeof(size_t));
    if (!from || !to) {
        free(from); free(to);
        free_matrix(working, nrow); free_matrix(orig, nrow);
        return -2;
    }

    size_t kk = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            if (working[i][j] == 1) { from[kk] = i; to[kk] = j; kk++; }

    scores[0] = 1.0;
    size_t index = 1;
    uint64_t state = make_seed(seed);
    int ret = 0;

    if (MAXITER == 0) {
        for (size_t n = 0; n < max_iter; n++) {
            if (verbose) load_bar(n, max_iter);
            size_t r1 = (size_t)(xrand(&state) * e);
            size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);
            size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];
            if (a != c && d != b && working[a][d] == 0 && working[c][b] == 0) {
                to[r1] = d; to[r2] = b;
                working[a][d] = working[c][b] = 1;
                working[a][b] = working[c][d] = 0;
            }
            if (n % step == 0)
                scores[index++] = jaccard_bipartite(working, orig, nrow, ncol, e);
        }
    } else {
        size_t n = 0, t = 0;
        while (n < max_iter) {
            if (verbose) load_bar(n, max_iter);
            size_t r1 = (size_t)(xrand(&state) * e);
            size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);
            size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];
            if (a != c && d != b && working[a][d] == 0 && working[c][b] == 0) {
                to[r1] = d; to[r2] = b;
                working[a][d] = working[c][b] = 1;
                working[a][b] = working[c][d] = 0;
                n++;
                if (n % step == 0)
                    scores[index++] = jaccard_bipartite(working, orig, nrow, ncol, e);
            }
            if (t++ > MAXITER) { ret = -1; break; }
        }
    }

    if (verbose) fprintf(stderr, "\nDONE\n");
    free(from); free(to);
    free_matrix(working, nrow); free_matrix(orig, nrow);
    return ret < 0 ? ret : (int)index;
}

/* ================================================================
   PUBLIC API — DENSE UNDIRECTED
   ================================================================ */

int bw_rewire_undirected(int16_t *flat, size_t ncol, size_t nrow,
                         size_t max_iter, int verbose, size_t MAXITER,
                         uint64_t seed) {
    short **m = flat_to_matrix(flat, nrow, ncol);
    if (!m) return -2;

    size_t e = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            if (m[i][j] == 1) e++;
    e /= 2;

    size_t *from = malloc(e * sizeof(size_t));
    size_t *to   = malloc(e * sizeof(size_t));
    if (!from || !to) {
        free(from); free(to); free_matrix(m, nrow);
        return -2;
    }

    size_t kk = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < i; j++)
            if (m[i][j] == 1) { from[kk] = i; to[kk] = j; kk++; }

    uint64_t state = make_seed(seed);
    int result = 0;

#define DO_UNDIRECTED_REWIRE(count_successful)                              \
    do {                                                                    \
        size_t r1 = (size_t)(xrand(&state) * e);                          \
        size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1==r2);\
        size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];       \
        if (a != c && b != d && a != d && c != b &&                        \
            ((m[a][d]==0 && m[c][b]==0) || (m[a][c]==0 && m[d][b]==0))) { \
            int both = (m[a][d]==0 && m[c][b]==0 && m[a][c]==0 && m[d][b]==0); \
            int path1 = both ? (xrand(&state) >= 0.5) : (m[a][d]==0 && m[c][b]==0); \
            if (path1) {                                                    \
                m[a][d]=m[d][a]=m[c][b]=m[b][c]=1;                        \
                m[a][b]=m[b][a]=m[c][d]=m[d][c]=0;                        \
                to[r1]=d; to[r2]=b;                                         \
            } else {                                                         \
                m[a][c]=m[c][a]=m[d][b]=m[b][d]=1;                        \
                m[a][b]=m[b][a]=m[c][d]=m[d][c]=0;                        \
                to[r1]=c; from[r2]=b; to[r2]=d;                            \
            }                                                               \
            if (count_successful) { (void)0; } /* marker */                 \
        }                                                                   \
    } while(0)

    if (MAXITER == 0) {
        for (size_t n = 0; n < max_iter; n++) {
            if (verbose) load_bar(n, max_iter);
            DO_UNDIRECTED_REWIRE(0);
        }
    } else {
        size_t n = 0, t = 0;
        /* We need to track successful rewirings separately */
        while (n < max_iter) {
            if (verbose) load_bar(n, max_iter);
            size_t r1 = (size_t)(xrand(&state) * e);
            size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);
            size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];
            if (a != c && b != d && a != d && c != b &&
                ((m[a][d]==0 && m[c][b]==0) || (m[a][c]==0 && m[d][b]==0))) {
                int both = (m[a][d]==0 && m[c][b]==0 && m[a][c]==0 && m[d][b]==0);
                int path1 = both ? (xrand(&state) >= 0.5) : (m[a][d]==0 && m[c][b]==0);
                if (path1) {
                    m[a][d]=m[d][a]=m[c][b]=m[b][c]=1;
                    m[a][b]=m[b][a]=m[c][d]=m[d][c]=0;
                    to[r1]=d; to[r2]=b;
                } else {
                    m[a][c]=m[c][a]=m[d][b]=m[b][d]=1;
                    m[a][b]=m[b][a]=m[c][d]=m[d][c]=0;
                    to[r1]=c; from[r2]=b; to[r2]=d;
                }
                n++;
            }
            if (t++ > MAXITER) { result = -1; break; }
        }
    }

#undef DO_UNDIRECTED_REWIRE

    if (verbose) fprintf(stderr, "\nDONE\n");
    matrix_to_flat(m, flat, nrow, ncol);
    free(from); free(to);
    free_matrix(m, nrow);
    return result;
}

int bw_analysis_undirected(int16_t *flat, size_t ncol, size_t nrow,
                           double *scores, size_t step, size_t max_iter,
                           int verbose, size_t MAXITER, uint64_t seed) {
    short **orig = flat_to_matrix(flat, nrow, ncol);
    if (!orig) return -2;
    short **working = copy_matrix(orig, nrow, ncol);
    if (!working) { free_matrix(orig, nrow); return -2; }

    size_t e = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < ncol; j++)
            if (orig[i][j] == 1) e++;
    e /= 2;

    size_t *from = malloc(e * sizeof(size_t));
    size_t *to   = malloc(e * sizeof(size_t));
    if (!from || !to) {
        free(from); free(to);
        free_matrix(working, nrow); free_matrix(orig, nrow);
        return -2;
    }

    size_t kk = 0;
    for (size_t i = 0; i < nrow; i++)
        for (size_t j = 0; j < i; j++)
            if (working[i][j] == 1) { from[kk] = i; to[kk] = j; kk++; }

    scores[0] = 1.0;
    size_t index = 1;
    uint64_t state = make_seed(seed);
    int ret = 0;

    /* Helper macro for one rewire attempt; sets 'rewired' to 1 if successful */
#define TRY_REWIRE_UNDIRECTED(rewired)                                          \
    do {                                                                         \
        size_t r1 = (size_t)(xrand(&state) * e);                                \
        size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);  \
        size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];             \
        rewired = 0;                                                             \
        if (a != c && b != d && a != d && c != b &&                             \
            ((working[a][d]==0 && working[c][b]==0) ||                           \
             (working[a][c]==0 && working[d][b]==0))) {                          \
            int both = (working[a][d]==0 && working[c][b]==0 &&                  \
                        working[a][c]==0 && working[d][b]==0);                   \
            int path1 = both ? (xrand(&state) >= 0.5)                           \
                             : (working[a][d]==0 && working[c][b]==0);           \
            if (path1) {                                                         \
                working[a][d]=working[d][a]=working[c][b]=working[b][c]=1;      \
                working[a][b]=working[b][a]=working[c][d]=working[d][c]=0;      \
                to[r1]=d; to[r2]=b;                                              \
            } else {                                                              \
                working[a][c]=working[c][a]=working[d][b]=working[b][d]=1;      \
                working[a][b]=working[b][a]=working[c][d]=working[d][c]=0;      \
                to[r1]=c; from[r2]=b; to[r2]=d;                                 \
            }                                                                    \
            rewired = 1;                                                         \
        }                                                                        \
    } while(0)

    if (MAXITER == 0) {
        for (size_t n = 0; n < max_iter; n++) {
            if (verbose) load_bar(n, max_iter);
            int rewired;
            TRY_REWIRE_UNDIRECTED(rewired);
            if (n % step == 0)
                scores[index++] = jaccard_undirected(working, orig, nrow, ncol, e);
        }
    } else {
        size_t n = 0, t = 0;
        while (n < max_iter) {
            if (verbose) load_bar(n, max_iter);
            int rewired;
            TRY_REWIRE_UNDIRECTED(rewired);
            if (rewired) {
                n++;
                if (n % step == 0)
                    scores[index++] = jaccard_undirected(working, orig, nrow, ncol, e);
            }
            if (t++ > MAXITER) { ret = -1; break; }
        }
    }

#undef TRY_REWIRE_UNDIRECTED

    if (verbose) fprintf(stderr, "\nDONE\n");
    free(from); free(to);
    free_matrix(working, nrow); free_matrix(orig, nrow);
    return ret < 0 ? ret : (int)index;
}

/* ================================================================
   PUBLIC API — SPARSE BIPARTITE
   ================================================================ */

int bw_rewire_sparse_bipartite(size_t *from, size_t *to,
                               size_t nc, size_t nr,
                               size_t max_iter, size_t e,
                               int verbose, size_t MAXITER,
                               uint64_t seed) {
    (void)nc;
    /* Build CSR-like index: index[g]..index[g+1] = range of edges in row g */
    size_t *index = malloc((nr + 1) * sizeof(size_t));
    size_t *pos   = malloc(e * sizeof(size_t));
    if (!index || !pos) {
        free(index); free(pos);
        return -2;
    }

    index[0] = 0;
    pos[0] = 0;
    size_t kk = 1, g = 0;
    for (size_t i = 1; i < e; i++) {
        if (from[i] != from[i - 1]) {
            index[kk++] = i;
            g++;
        }
        pos[i] = g;
    }
    index[nr] = e;

    uint64_t state = make_seed(seed);
    int result = 0;

    if (MAXITER == 0) {
        for (size_t n = 0; n < max_iter; n++) {
            if (verbose) load_bar(n, max_iter);
            size_t r1 = (size_t)(xrand(&state) * e);
            size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);
            size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];
            if (a != c && d != b && check_sparse(pos, to, index, r1, b, r2, d)) {
                to[r1] = d; to[r2] = b;
            }
        }
    } else {
        size_t n = 0, t = 0;
        while (n < max_iter) {
            if (verbose) load_bar(n, max_iter);
            size_t r1 = (size_t)(xrand(&state) * e);
            size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);
            size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];
            if (a != c && d != b && check_sparse(pos, to, index, r1, b, r2, d)) {
                to[r1] = d; to[r2] = b;
                n++;
            }
            if (t++ > MAXITER) { result = -1; break; }
        }
    }

    if (verbose) fprintf(stderr, "\nDONE\n");
    free(index); free(pos);
    return result;
}

/* ================================================================
   PUBLIC API — SPARSE UNDIRECTED
   ================================================================ */

int bw_rewire_sparse(size_t *from, size_t *to, size_t *degree,
                     size_t nc, size_t nr,
                     size_t max_iter, size_t e,
                     int verbose, size_t MAXITER,
                     uint64_t seed) {
    (void)nc;
    /* Build adjacency list representation for fast edge lookup */
    short **adj = malloc(nr * sizeof(short *));
    if (!adj) return -2;
    for (size_t i = 0; i < nr; i++) {
        /* Extra slot at index degree[i] holds a fill counter (decreasing) */
        adj[i] = malloc((degree[i] + 1) * sizeof(short));
        if (!adj[i]) {
            for (size_t k = 0; k < i; k++) free(adj[k]);
            free(adj);
            return -2;
        }
        adj[i][degree[i]] = (short)degree[i];
    }

    for (size_t i = 0; i < e; i++) {
        size_t a = from[i], b = to[i];
        /* Fill from the end; counter in adj[a][degree[a]] tracks next slot */
        adj[a][degree[a] - (size_t)adj[a][degree[a]]] = (short)b;
        adj[a][degree[a]]--;
        adj[b][degree[b] - (size_t)adj[b][degree[b]]] = (short)a;
        adj[b][degree[b]]--;
    }

    uint64_t state = make_seed(seed);
    int result = 0;

#define TRY_SPARSE_UNDIRECTED(rewired)                                           \
    do {                                                                          \
        size_t r1 = (size_t)(xrand(&state) * e);                                 \
        size_t r2; do { r2 = (size_t)(xrand(&state) * e); } while (r1 == r2);   \
        size_t a = from[r1], b = to[r1], c = from[r2], d = to[r2];              \
        int ad = is_not(a,d,degree,adj), cb = is_not(c,b,degree,adj);            \
        int ac = is_not(c,a,degree,adj), bd = is_not(b,d,degree,adj);            \
        rewired = 0;                                                              \
        if (a!=c && b!=d && a!=d && c!=b && ((ad && cb) || (ac && bd))) {        \
            if (ad && cb && ac && bd) {                                           \
                if (xrand(&state) >= 0.5) {                                      \
                    to[r1]=d; to[r2]=b; sub1(a,b,c,d,degree,adj);               \
                } else {                                                          \
                    sub2(a,b,c,d,degree,adj);                                    \
                    to[r1]=c; from[r2]=b; to[r2]=d;                              \
                }                                                                 \
            } else if (ad && cb) {                                                \
                to[r1]=d; to[r2]=b; sub1(a,b,c,d,degree,adj);                  \
            } else {                                                              \
                sub2(a,b,c,d,degree,adj);                                        \
                to[r1]=c; from[r2]=b; to[r2]=d;                                 \
            }                                                                     \
            rewired = 1;                                                          \
        }                                                                         \
    } while(0)

    if (MAXITER == 0) {
        for (size_t n = 0; n < max_iter; n++) {
            if (verbose) load_bar(n, max_iter);
            int rewired;
            TRY_SPARSE_UNDIRECTED(rewired);
            (void)rewired;
        }
    } else {
        size_t n = 0, t = 0;
        while (n < max_iter) {
            if (verbose) load_bar(n, max_iter);
            int rewired;
            TRY_SPARSE_UNDIRECTED(rewired);
            if (rewired) n++;
            if (t++ > MAXITER) { result = -1; break; }
        }
    }

#undef TRY_SPARSE_UNDIRECTED

    if (verbose) fprintf(stderr, "\nDONE\n");
    for (size_t i = 0; i < nr; i++) free(adj[i]);
    free(adj);
    return result;
}
