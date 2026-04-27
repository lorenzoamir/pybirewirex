import math


def bound_bipartite(e: float, t: float, accuracy: float, exact: bool) -> int:
    """Iteration bound for bipartite rewiring.

    Args:
        e: number of edges
        t: total possible edges (nrow * ncol)
        accuracy: convergence threshold
        exact: if True use exact formula, otherwise approximation
    """
    ratio = e / t
    one_minus = 1.0 - ratio
    log_term = math.log(one_minus / accuracy)
    if exact:
        return math.ceil((e * one_minus * log_term) / 2.0)
    else:
        return math.ceil((e / (2.0 * one_minus)) * log_term)


def bound_undirected(e: float, t: float, accuracy: float, exact: bool) -> int:
    """Iteration bound for undirected rewiring.

    Args:
        e: number of edges (sum of upper triangle)
        t: n^2 / 2 where n is number of nodes
        accuracy: convergence threshold
        exact: if True use exact formula, otherwise density-dependent cubic
    """
    ratio = e / t
    one_minus = 1.0 - ratio
    if exact:
        log_term = math.log(one_minus / accuracy)
        return math.ceil((e * one_minus * log_term) / 2.0)
    else:
        d = ratio
        denom = 2.0 * d**3 - 6.0 * d**2 + 2.0 * d + 2.0
        log_term = math.log((1.0 - d) / accuracy)
        return math.ceil((e / denom) * log_term)
