"""Streams for T09. Each loader returns (X, y, info) in chronological order.

info: dict with n_features, n_classes, drift_points and the generator parameters
(saved in the run's .json). Add RBF / Agrawal / Elec2 loaders here (Pessoa 2)
and register them in STREAMS.
"""
from __future__ import annotations

import numpy as np


def aba(n: int, seed: int):
    """Recurring drift A-B-A (protocol definition).
    A: x ~ U[0,0.6]^2, y = 1[x0 > 0.3]
    B: x ~ U[0.4,1]^2, y = 1[x1 > 0.7]
    A returns. Changes at 1/3 and 2/3 of the stream."""
    rng = np.random.default_rng(seed)
    b1, b2 = n // 3, 2 * n // 3
    X = np.empty((n, 2))
    y = np.empty(n, dtype=int)
    for i in range(n):
        if i < b1 or i >= b2:
            X[i] = rng.uniform(0.0, 0.6, 2)
            y[i] = int(X[i, 0] > 0.3)
        else:
            X[i] = rng.uniform(0.4, 1.0, 2)
            y[i] = int(X[i, 1] > 0.7)
    info = {"n_features": 2, "n_classes": 2, "drift_points": [b1, b2],
            "generator": "aba", "params": {"A": "U[0,0.6]^2, y=1[x0>0.3]",
                                           "B": "U[0.4,1]^2, y=1[x1>0.7]"}}
    return X, y, info


STREAMS = {
    "aba": aba,
    # "rbf": rbf,          # [Pessoa 2]
    # "agrawal": agrawal,  # [Pessoa 2]
    # "elec2": elec2,      # [Pessoa 2]
}
