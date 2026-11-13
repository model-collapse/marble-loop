"""Protected copy of the original baseline, used as the timing reference."""
import math


def pairwise_distances(X):
    n = len(X)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for a, b in zip(X[i], X[j]):
                s += (a - b) ** 2
            out[i][j] = math.sqrt(s)
    return out
