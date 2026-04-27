"""
PyBiRewireX demo — mirrors the BiRewire R package vignette (BiRewire.Rnw).

Vignette sections reproduced:
  §birewire.analysis.bipartite   — Jaccard convergence, bound N
  §birewire.rewire.bipartite     — degree-preserving randomisation
  §birewire.similarity           — Jaccard index
  §birewire.analysis.undirected  — same workflow for undirected graphs
  §birewire.rewire.undirected
  §birewire.sampler.bipartite    — generate K null-model networks
  §birewire.visual.monitoring    — t-SNE of Markov chain (both graph types)

R vignette "Example" network:
    bipartite.random.game(n1=100, n2=40, p=0.2)   →  100×40, ~20 % density
    erdos.renyi.game(n=200, p=0.05, directed=F)    →  ~200-node undirected

Figures saved to scripts/output/:
    demo_analysis.png    — analogue of vignette analysis.pdf
    demo_monitoring.png  — analogue of vignette monitoring.pdf

Run:
    uv run python scripts/demo.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import t as t_dist
from pathlib import Path

import pybirewirex as pbr
from pybirewirex.similarity import jaccard
from pybirewirex._bounds import bound_bipartite

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)

sep = "=" * 60


# ── 1. Create networks ───────────────────────────────────────────────────────
# R vignette:
#   data(BRCA_binary_matrix)   ← real genomic BEM (757×9757)
#   ggg <- bipartite.random.game(n1=100, n2=40, p=0.2)
#   g.und <- erdos.renyi.game(directed=F, loops=F, n=1000, p.or.m=0.01)
#
# Python: synthetic equivalents at the same densities but smaller scale.
# ---------------------------------------------------------------------------

print(sep)
print("1. Create networks")
print(sep)

# 100×40 bipartite at 20% density  (= vignette bipartite.random.game)
bp = (rng.random((100, 40)) < 0.20).astype(np.int16)
e_bp = int(bp.sum())
print(f"  Bipartite  {bp.shape[0]}×{bp.shape[1]},  {e_bp} edges "
      f"({e_bp / bp.size:.2%} density)")

# 200-node undirected Erdős–Rényi at 5% density
# Vignette uses n=1000, p=0.01; we scale down for speed.
_full  = (rng.random((200, 200)) < 0.05).astype(np.int16)
_upper = np.triu(_full, k=1)
und    = (_upper + _upper.T).clip(0, 1).astype(np.int16)
e_und  = int(und.sum()) // 2
print(f"  Undirected {und.shape[0]} nodes,  {e_und} edges "
      f"({e_und / (und.shape[0]*(und.shape[0]-1)//2):.2%} density)")


# ── 2. Bipartite: compute analytical bound N ─────────────────────────────────
# R vignette:
#   N <- birewire.analysis.bipartite(get.incidence(ggg), max.iter=2, step=1)$N
#   (quick pass just to obtain the bound; analysis itself not displayed)
# ---------------------------------------------------------------------------

print()
print(sep)
print("2. Bipartite: analytical bound N")
print(sep)

N_bp = bound_bipartite(e_bp, bp.size, accuracy=0.00005, exact=False)
print(f"  N = {N_bp}  (formula: ceil( e/(2-2d) · ln((1-d)/δ) ), δ=5×10⁻⁵ )")


# ── 3. Bipartite: convergence analysis ──────────────────────────────────────
# R vignette:
#   res <- birewire.analysis.bipartite(
#               get.incidence(ggg), max.iter=10*N, n.networks=10)
#   (display=TRUE produces analysis.pdf)
# ---------------------------------------------------------------------------

print()
print(sep)
print("3. Bipartite: Jaccard convergence analysis")
print(sep)

MAX_ITER_BP = 10 * N_bp
STEP_BP     = 10   # R default; gives ~N points for a smooth curve
N_NETWORKS  = 10

print(f"  max_iter={MAX_ITER_BP}  step={STEP_BP}  n_networks={N_NETWORKS}")
bp_result = pbr.analysis_bipartite(
    bp,
    step       = STEP_BP,
    max_iter   = MAX_ITER_BP,
    n_networks = N_NETWORKS,
    accuracy   = 0.00005,
    exact      = False,
    verbose    = False,
    seed       = SEED,
)
print(f"  scores shape: {bp_result.scores.shape}  (n_networks × n_steps)")

# ── 4. Undirected: bound + convergence analysis ──────────────────────────────
# R vignette:
#   g.und <- erdos.renyi.game(n=1000, p.or.m=0.01, directed=F, loops=F)
#   m.und <- get.adjacency(g.und, sparse=FALSE)
#   scores.und <- birewire.analysis.undirected(m.und, step=100, max.iter=max,
#                                               n.networks=5, verbose=FALSE)
# ---------------------------------------------------------------------------

print()
print(sep)
print("4. Undirected: Jaccard convergence analysis")
print(sep)

t_und    = und.shape[0] * (und.shape[0] - 1) // 2
N_und    = bound_bipartite(e_und, t_und, accuracy=0.00005, exact=False)
MAX_ITER_UND = 10 * N_und
STEP_UND     = 10

print(f"  N = {N_und}  max_iter={MAX_ITER_UND}  step={STEP_UND}  n_networks=5")
und_result = pbr.analysis_undirected(
    und,
    step       = STEP_UND,
    max_iter   = MAX_ITER_UND,
    n_networks = 5,
    accuracy   = 0.00005,
    exact      = False,
    verbose    = False,
    seed       = SEED,
)
print(f"  scores shape: {und_result.scores.shape}")

# ── 5. Plot analysis figure (analogue of analysis.pdf) ────────────────────────
# Matches R output exactly:
#   par(mfrow=c(2,1))
#   plot(step*x, mean, ...)           ← linear scale, top panel
#   plot(step*x, mean, ..., log='xy') ← log-log scale, bottom panel
#   polygon(...)  col='grey80'        ← CI band
#   lines(mean)   col='blue', lwd=2   ← mean line
#   abline(v=N)   col='red'           ← bound line
#   legend: "Mean JI" / "C.I." / "Bound"
# R uses x=seq(1, length.out=n_steps), so x-axis starts at step (not 0).
# ---------------------------------------------------------------------------

def plot_analysis_panels(result, axes):
    scores = result.scores
    n_net  = scores.shape[0]
    xs     = np.arange(1, scores.shape[1] + 1) * result.step
    mean   = scores.mean(axis=0)
    se     = scores.std(axis=0, ddof=1) / np.sqrt(n_net)
    tval   = t_dist.ppf(0.975, df=n_net - 1) if n_net > 1 else 1.96
    sup    = mean + tval * se
    inf_   = mean - tval * se

    for ax, logscale in zip(axes, [False, True]):
        ax.fill_between(xs, inf_, sup, color="#CCCCCC", label="C.I.")
        ax.plot(xs, mean, color="blue", linewidth=2, label="Mean JI")
        ax.axvline(result.N, color="red", linewidth=1, label="Bound")
        ax.set_xlabel("Switching steps", fontsize=10)
        ax.set_ylabel("Jaccard Index", fontsize=10)
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("Jaccard index (JI) over time (log-log scale)",
                         fontsize=10, fontweight="bold")
            ax.legend(loc="lower left", fontsize=8)
        else:
            ax.set_title("Jaccard index (JI) over time",
                         fontsize=10, fontweight="bold")
            ax.legend(loc="upper right", fontsize=8)


fig_an, axes_an = plt.subplots(2, 1, figsize=(7, 8))
plot_analysis_panels(bp_result, axes_an)
fig_an.tight_layout()
fig_an.savefig(OUT / "demo_analysis.png", dpi=150, bbox_inches="tight")
print(f"\n  → {OUT / 'demo_analysis.png'}")


# ── 6. Rewiring + similarity ──────────────────────────────────────────────────
# R vignette:
#   m2  <- birewire.rewire.bipartite(BRCA_binary_matrix, verbose=FALSE)
#   g2  <- birewire.rewire.bipartite(g, verbose=FALSE)        # igraph input
#   sc  <- birewire.similarity(BRCA_binary_matrix, m2)
#   m2.und <- birewire.rewire.undirected(m.und, verbose=FALSE)
# ---------------------------------------------------------------------------

print()
print(sep)
print("5. Rewiring + Jaccard similarity")
print(sep)

bp_rewired  = pbr.rewire_bipartite(bp,  verbose=False, seed=SEED)
und_rewired = pbr.rewire_undirected(und, verbose=False, seed=SEED)

j_bp  = jaccard(bp, bp_rewired)
j_und = jaccard(und, und_rewired)

print(f"  Bipartite  row sums preserved: "
      f"{np.array_equal(bp.sum(1), bp_rewired.sum(1))}")
print(f"  Bipartite  col sums preserved: "
      f"{np.array_equal(bp.sum(0), bp_rewired.sum(0))}")
print(f"  Bipartite  Jaccard(original, rewired) = {j_bp:.4f}")
print()
print(f"  Undirected degrees preserved:  "
      f"{np.array_equal(und.sum(0), und_rewired.sum(0))}")
print(f"  Undirected symmetric:          "
      f"{np.array_equal(und_rewired, und_rewired.T)}")
print(f"  Undirected Jaccard(original, rewired) = {j_und:.4f}")


# ── 7. Sampler ────────────────────────────────────────────────────────────────
# R vignette:
#   birewire.sampler.bipartite(ggg, K=10000, path="TESTBIREWIREBIPARTITE")
#
# Python: sequential rewiring with N SS between samples — equivalent to the
# R sampler (each saved network is N SS beyond the previous one).
# ---------------------------------------------------------------------------

print()
print(sep)
print("6. Sampler: draw K networks from the null model")
print(sep)

K_SAMPLES = 5
sampled   = []
current   = bp.copy()
for k in range(K_SAMPLES):
    current = pbr.rewire_bipartite(current, verbose=False, seed=SEED + k + 1)
    sampled.append(current.copy())

row_ok = all(np.array_equal(s.sum(1), bp.sum(1)) for s in sampled)
col_ok = all(np.array_equal(s.sum(0), bp.sum(0)) for s in sampled)
print(f"  Generated {K_SAMPLES} null-model networks")
print(f"  All row sums preserved: {row_ok}")
print(f"  All col sums preserved: {col_ok}")
pairwise = [jaccard(sampled[i], sampled[j])
            for i in range(K_SAMPLES) for j in range(i+1, K_SAMPLES)]
print(f"  Pairwise Jaccard among samples: "
      f"mean={np.mean(pairwise):.3f}  min={np.min(pairwise):.3f}  "
      f"max={np.max(pairwise):.3f}")


# ── 8. Monitoring: t-SNE of Markov chain ────────────────────────────────────
# R vignette (monitoring.pdf):
#   tsne = birewire.visual.monitoring.bipartite(
#               ggg, display=T, n.networks=75,
#               sequence=c(1,10,200,1000,"n",50000), ncol=3, perplexity=10)
#
# R implementation (BiRewire.R lines 752-814):
#   for each k in sequence:
#     tot = [data]
#     for j in 2:n.networks:
#       data_tmp = birewire.rewire.bipartite(data_tmp, max.iter=k)  ← sequential
#       tot[[j]] = data_tmp
#       for l in 1:(j-1):
#         m[l,j] = m[j,l] = 1 - birewire.similarity(tot[[l]], tot[[j]])
#     Rtsne(m, perplexity=perplexity, check_duplicates=FALSE)   ← NO is_distance
#
# Rtsne pipeline (no is_distance=TRUE):
#   treat m as feature matrix → normalize (global max-abs) → PCA(50) → TSNE(euclidean)
#
# Python differences:
#   • PRNG: R=Mersenne Twister, Python=xorshift64 — different trajectories,
#     same stationary distribution.
#   • "n" in sequence: Python resolves bound_bipartite() before calling rewire.
#   • t-SNE init: R uses "random" by default; we match with init="random".
# ---------------------------------------------------------------------------

print()
print(sep)
print("7. Monitoring: t-SNE visualisation of the Markov chain")
print(sep)

# Matches R vignette output (monitoring.pdf):
#   birewire.visual.monitoring.bipartite(ggg, display=T, n.networks=75,
#       sequence=c(1,10,200,1000,"n",50000), ncol=3, perplexity=10)
# → 6 panels in a 2×3 grid, bipartite only.
N_MON  = 75
PERP   = 10
SEQ    = [1, 10, 200, 1000, N_bp, 50000]
LABS   = ["1", "10", "200", "1000", "N", "50000"]


def collect_chain(start, interval, n_samples, base_seed):
    """Sequential Markov chain walk — matches R's inner loop."""
    nets    = [start.copy()]
    current = start.copy()
    for i in range(n_samples - 1):
        current = pbr.rewire_bipartite(
            current, max_iter=interval, verbose=False, seed=base_seed + i)
        nets.append(current.copy())
    return nets


def pairwise_dist(nets):
    n = len(nets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - jaccard(nets[i], nets[j])
    return D


def embed_tsne(D, perplexity, seed):
    """Rtsne-faithful: normalize → PCA(50) → t-SNE(euclidean)."""
    D_n   = D / (np.abs(D).max() + 1e-12)
    n_pca = min(50, D_n.shape[1] - 1)
    D_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(D_n)
    return TSNE(n_components=2, metric="euclidean", perplexity=perplexity,
                random_state=seed, init="random", max_iter=1000
                ).fit_transform(D_pca)


def scatter_panel(ax, emb, label, n):
    cmap   = plt.cm.coolwarm
    norm   = Normalize(vmin=0, vmax=n - 1)
    colors = cmap(norm(np.arange(n)))
    ax.scatter(emb[1:, 0], emb[1:, 1], c=colors[1:], s=10, linewidths=0)
    ax.scatter(emb[0, 0], emb[0, 1], color=colors[0], s=50,
               edgecolors="black", linewidths=0.8)
    ax.text(emb[0, 0], emb[0, 1], " start", fontsize=8)
    ax.set_title(f"k= {label}", fontsize=10, fontweight="bold", pad=3)
    ax.set_xlabel("A.U.", fontsize=8)
    ax.set_ylabel("A.U.", fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


print(f"  n_networks={N_MON}, perplexity={PERP}, sequence={LABS}")
embeddings = []
for k, (iv, lbl) in enumerate(zip(SEQ, LABS)):
    print(f"    k={lbl:<7} … ", end="", flush=True)
    nets = collect_chain(bp, iv, N_MON, base_seed=SEED + k * 10000)
    D    = pairwise_dist(nets)
    embeddings.append(embed_tsne(D, PERP, SEED))
    print("done")

# ── Plot monitoring figure: 2 rows × 3 cols (ncol=3 as in R vignette) ────────
fig_mon, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, emb, lbl in zip(axes.ravel(), embeddings, LABS):
    scatter_panel(ax, emb, lbl, N_MON)
fig_mon.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.08,
                        hspace=0.45, wspace=0.35)
fig_mon.savefig(OUT / "demo_monitoring.png", dpi=150, bbox_inches="tight")
print(f"\n  → {OUT / 'demo_monitoring.png'}")

print()
print("Done.")
