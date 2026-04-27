"""
pybirewirex — Python port of the BiRewire network rewiring algorithms.
"""

from pybirewirex._core import _C_AVAILABLE
from pybirewirex.bipartite import AnalysisResult, analysis_bipartite, rewire_bipartite
from pybirewirex.undirected import analysis_undirected, rewire_undirected

__all__ = [
    "_C_AVAILABLE",
    "rewire_bipartite",
    "analysis_bipartite",
    "rewire_undirected",
    "analysis_undirected",
    "AnalysisResult",
]
