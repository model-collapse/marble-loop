"""Editable: compute the full matrix of pairwise Euclidean distances between rows of X."""
import math


def pairwise_distances(X):
    """X: list of n points (lists of d floats). Returns an n x n list of lists."""
    n = len(X)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for a, b in zip(X[i], X[j]):
                s += (a - b) ** 2
            out[i][j] = math.sqrt(s)
    return out
