# Export reference data from R's BiRewire package for regression testing.
#
# Run this script to regenerate tests/reference_data/r_*.csv files:
#   Rscript tests/export_references.R
#
# NOTE: Python uses xorshift64 PRNG; R uses Mersenne Twister via set.seed().
# Rewired matrix entries differ between R and Python for the same seed, but
# BOTH must preserve degree sequences exactly. Regression tests in
# tests/test_regression.py compare degree sequences and convergence properties,
# not raw matrix entries.

library(BiRewire)

OUTDIR <- file.path(dirname(normalizePath("tests/export_references.R")), "reference_data")
if (!dir.exists(OUTDIR)) dir.create(OUTDIR, recursive = TRUE)

SEED <- 42

# ---- bipartite input (4×5) -------------------------------------------

# Same matrix as Python fixture: [[1,0,1,0,1],[0,1,0,1,0],[1,1,0,0,1],[0,0,1,1,0]]
# R fills column-major, so provide columns: c(col1, col2, col3, col4, col5)
bip_input <- matrix(
  c(1, 0, 1, 0,  0, 1, 1, 0,  1, 0, 0, 1,  0, 1, 0, 1,  1, 0, 1, 0),
  nrow = 4, ncol = 5
)
write.csv(bip_input, file.path(OUTDIR, "r_bipartite_input.csv"), row.names = FALSE)

# ---- rewired bipartite ---------------------------------------------------

set.seed(SEED)
bip_rewired <- birewire.rewire.bipartite(bip_input, MAXITER = 500, verbose = FALSE)
write.csv(bip_rewired, file.path(OUTDIR, "r_bipartite_rewired.csv"), row.names = FALSE)
write.csv(
  data.frame(row_sums = rowSums(bip_input)),
  file.path(OUTDIR, "r_bipartite_row_sums.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(col_sums = colSums(bip_input)),
  file.path(OUTDIR, "r_bipartite_col_sums.csv"),
  row.names = FALSE
)

# ---- bipartite analysis (50 networks, step=10, MAXITER=500) -------------

set.seed(SEED)
bip_analysis <- birewire.analysis.bipartite(
  bip_input, step = 10, MAXITER = 500, verbose = FALSE
)
cat("R bipartite N:", bip_analysis$N, "\n")
cat("R bipartite scores dim:", dim(bip_analysis$data), "\n")
write.csv(bip_analysis$data, file.path(OUTDIR, "r_bipartite_scores.csv"), row.names = FALSE)
write.csv(
  data.frame(N = bip_analysis$N),
  file.path(OUTDIR, "r_bipartite_N.csv"),
  row.names = FALSE
)

# ---- undirected input (5×5) -------------------------------------------

und_input <- matrix(
  c(0, 1, 1, 0, 0,  1, 0, 0, 1, 1,  1, 0, 0, 1, 0,  0, 1, 1, 0, 1,  0, 1, 0, 1, 0),
  nrow = 5, ncol = 5, byrow = TRUE
)
write.csv(und_input, file.path(OUTDIR, "r_undirected_input.csv"), row.names = FALSE)

# ---- rewired undirected -------------------------------------------------

set.seed(SEED)
und_rewired <- birewire.rewire.undirected(und_input, MAXITER = 500, verbose = FALSE)
write.csv(und_rewired, file.path(OUTDIR, "r_undirected_rewired.csv"), row.names = FALSE)
write.csv(
  data.frame(degree = rowSums(und_input)),
  file.path(OUTDIR, "r_undirected_degree.csv"),
  row.names = FALSE
)

# ---- undirected analysis (50 networks, step=10, MAXITER=500) ------------

set.seed(SEED)
und_analysis <- birewire.analysis.undirected(
  und_input, step = 10, MAXITER = 500, verbose = FALSE
)
cat("R undirected N:", und_analysis$N, "\n")
cat("R undirected scores dim:", dim(und_analysis$data), "\n")
write.csv(und_analysis$data, file.path(OUTDIR, "r_undirected_scores.csv"), row.names = FALSE)
write.csv(
  data.frame(N = und_analysis$N),
  file.path(OUTDIR, "r_undirected_N.csv"),
  row.names = FALSE
)

cat("\nReference CSVs written to", OUTDIR, "\n")
