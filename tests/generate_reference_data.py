"""Generate .npy reference files for regression tests.

Run once with a known-good version of the C backend:
    uv run python tests/generate_reference_data.py

Outputs are committed to tests/reference_data/ so CI runs without re-generation.
"""

from pathlib import Path

import numpy as np

OUTDIR = Path(__file__).parent / "reference_data"
OUTDIR.mkdir(exist_ok=True)

SEED = 42

# ---- synthetic bipartite input (4×5) ------------------------------------

bip_input = np.array(
    [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ],
    dtype=np.int16,
)
np.save(OUTDIR / "bipartite_input.npy", bip_input)

# ---- rewired bipartite --------------------------------------------------

from pybirewirex.bipartite import rewire_bipartite, analysis_bipartite  # noqa: E402

bip_rewired = rewire_bipartite(bip_input, max_iter=500, verbose=False, seed=SEED)
np.save(OUTDIR / "bipartite_rewired.npy", bip_rewired)

# ---- bipartite analysis trajectory (5 networks, step=10) ---------------

bip_result = analysis_bipartite(
    bip_input, step=10, max_iter=500, n_networks=5, verbose=False, seed=SEED
)
np.save(OUTDIR / "bipartite_scores.npy", bip_result.scores)
np.save(OUTDIR / "bipartite_N.npy", np.array([bip_result.N], dtype=np.int64))

# ---- synthetic undirected input (5×5) -----------------------------------

und_input = np.array(
    [
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
    ],
    dtype=np.int16,
)
np.save(OUTDIR / "undirected_input.npy", und_input)

# ---- rewired undirected -------------------------------------------------

from pybirewirex.undirected import rewire_undirected, analysis_undirected  # noqa: E402

und_rewired = rewire_undirected(und_input, max_iter=500, verbose=False, seed=SEED)
np.save(OUTDIR / "undirected_rewired.npy", und_rewired)

# ---- undirected analysis trajectory (5 networks, step=10) --------------

und_result = analysis_undirected(
    und_input, step=10, max_iter=500, n_networks=5, verbose=False, seed=SEED
)
np.save(OUTDIR / "undirected_scores.npy", und_result.scores)
np.save(OUTDIR / "undirected_N.npy", np.array([und_result.N], dtype=np.int64))

print("Reference data written to", OUTDIR)
for p in sorted(OUTDIR.glob("*.npy")):
    arr = np.load(p)
    print(f"  {p.name}: shape={arr.shape} dtype={arr.dtype}")
