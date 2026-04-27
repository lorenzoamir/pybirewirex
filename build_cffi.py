import sys

from cffi import FFI

ffi = FFI()

ffi.set_source(
    "pybirewirex._birewire",
    r"""
    #include "birewire_core.h"
    """,
    sources=["c_src/birewire_core.c"],
    include_dirs=["c_src"],
    extra_compile_args=["-O2", "-std=c99", "-Wall"] if sys.platform != "win32" else ["/O2"],
    libraries=["m"] if sys.platform != "win32" else [],
)

ffi.cdef("""
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

    int bw_rewire_sparse_bipartite(size_t *from, size_t *to,
                                   size_t nc, size_t nr,
                                   size_t max_iter, size_t e,
                                   int verbose, size_t MAXITER,
                                   uint64_t seed);

    int bw_rewire_sparse(size_t *from, size_t *to, size_t *degree,
                         size_t nc, size_t nr,
                         size_t max_iter, size_t e,
                         int verbose, size_t MAXITER,
                         uint64_t seed);
""")

if __name__ == "__main__":
    ffi.compile(verbose=True)
