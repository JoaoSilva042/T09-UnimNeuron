"""Streams for T09. Each loader returns (X, y, info) in chronological order.

info: dict with n_features, n_classes, drift_points and the generator parameters
(saved in the run's .json). Add RBF / Agrawal / Elec2 loaders here (Pessoa 2)
and register them in STREAMS.
"""
from __future__ import annotations

import numpy as np

import itertools

from river.datasets import synth, Elec2


def _minmax(X):
    """Igual ao baseline (min_max_scale): min/max por feature sobre as N
    observacoes carregadas; denominador 0 -> 1."""
    x_min = X.min(axis=0)
    denom = X.max(axis=0) - x_min
    denom[denom == 0.0] = 1.0
    return (X - x_min) / denom


def _encode_labels(y_raw):
    """Labels -> inteiros 0..C-1 pela ordem de primeira aparicao (como o baseline)."""
    mapping = {v: i for i, v in enumerate(dict.fromkeys(y_raw))}
    return np.array([mapping[v] for v in y_raw], dtype=int)


def _collect(stream, n):
    """Le as primeiras n observacoes de um stream da River."""
    pairs = list(itertools.islice(stream, n))
    feats = list(pairs[0][0].keys())
    X = np.array([[x[f] for f in feats] for x, _ in pairs], dtype=float)
    y = _encode_labels([y for _, y in pairs])
    return X, y, feats

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


def rbf(n: int, seed: int, change_speed: float = 0.001, n_features: int = 10,
        n_centroids: int = 50, n_drift_centroids: int = 50):
    """RandomRBFDrift (River): drift continuo em P(x) e P(y|x).
    Por omissao a River tem change_speed=0.0 (SEM drift) e seeds None;
    drift explicito e seeds fixas."""
    stream = synth.RandomRBFDrift(
        seed_model=seed, seed_sample=seed, n_classes=2,
        n_features=n_features, n_centroids=n_centroids,
        change_speed=change_speed, n_drift_centroids=n_drift_centroids)
    X, y, feats = _collect(stream, n)
    info = {"n_features": int(X.shape[1]), "n_classes": int(y.max()) + 1,
            "drift_points": [],   # drift continuo: nao ha ponto de drift
            "generator": "rbf",
            "params": {"change_speed": change_speed, "n_centroids": n_centroids,
                       "n_drift_centroids": n_drift_centroids,
                       "seed_model": seed, "seed_sample": seed}}
    return _minmax(X), y, info

def agrawal(n: int, seed: int, f_before: int = 0, f_after: int = 4,
            width: int = 1000):
    """ConceptDrift_Agrawal: controlo negativo (so P(y|x) muda, P(x) nao).
    O baseline original usa Agrawal() com a funcao por omissao (0) nos dois
    lados, ou seja, SEM drift real; as funcoes sao explicitas (0 -> 4).
    O drift fica a meio do stream (position = n // 2)."""
    position = n // 2
    stream = synth.ConceptDriftStream(
        stream=synth.Agrawal(classification_function=f_before, seed=seed),
        drift_stream=synth.Agrawal(classification_function=f_after, seed=seed),
        seed=seed, position=position, width=width)
    X, y, feats = _collect(stream, n)
    info = {"n_features": int(X.shape[1]), "n_classes": int(y.max()) + 1,
            "drift_points": [position],
            "generator": "agrawal",
            "params": {"f_before": f_before, "f_after": f_after,
                       "position": position, "width": width, "seed": seed}}
    return _minmax(X), y, info

def elec2(n: int, seed: int = 0):
    """Elec2 (dados reais, River): prefixo cronologico das primeiras n observacoes.
    O stream e fixo e o modelo e deterministico, por isso a seed e ignorada
    (existe so para ter a mesma interface dos outros loaders)."""
    X, y, feats = _collect(Elec2(), n)
    info = {"n_features": int(X.shape[1]), "n_classes": int(y.max()) + 1,
            "drift_points": [],   # sem pontos de drift conhecidos
            "generator": "elec2",
            "params": {"prefix": n, "features": feats, "seed_ignored": True}}
    return _minmax(X), y, info

STREAMS = {
    "aba": aba,
    "rbf": rbf, 
    "agrawal": agrawal,
    "elec2": elec2,
}
