"""Diagnóstico para as grelhas (T09).

Corre o B0 SEM pruning num stream de desenvolvimento e mede, observação a
observação, as quantidades de que dependem H, H0, k (S1/S2) e alpha, theta (S3).
O modelo é exatamente o B0: só observamos, nunca removemos regras.

Uso (a partir da raiz do repositório):
    uv run python -m t09.diagnostics --n 5000 --seed 100
"""
from __future__ import annotations

import argparse

import numpy as np

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe

from t09.streams import aba


# ---------------------------------------------------------------------------
# 1. Stream A-B-A (definição do protocolo)
# ---------------------------------------------------------------------------

def aba_stream(n: int, seed: int):
    """Same stream as t09.streams.aba (kept here for the diagnostic summary)."""
    X, y, info = aba(n, seed)
    return X, y, tuple(info["drift_points"])


def region(center: np.ndarray) -> str:
    """Zona onde está o centro da regra: só A, só B, ou sobreposição."""
    if np.all(center < 0.4):
        return "A"          # só o conceito A gera pontos aqui
    if np.all(center > 0.6):
        return "B"          # só o conceito B gera pontos aqui
    return "AB"


# ---------------------------------------------------------------------------
# 2. B0 com monitorização (sem pruning)
# ---------------------------------------------------------------------------

class B0Diagnostic(EvolvingNeuroFuzzyUnimSafe):
    """B0 original + registo das quantidades. Nunca remove regras."""

    def __init__(self, *args, alphas=(0.005, 0.01, 0.02, 0.05, 0.1),
                 beta=0.90, act_min_rel=0.30, dist_factor_rel=3.0,
                 snapshot_every=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.alphas = tuple(alphas)
        self.beta, self.act_min_rel, self.dist_factor_rel = beta, act_min_rel, dist_factor_rel
        self.snapshot_every = snapshot_every
        self.gaps = []          # (t, regiao, gap) de todas as reativações
        self.snapshots = []     # fotografias periódicas do estado das regras

    def _update_one(self, x, y):
        x = np.asarray(x, dtype=float).ravel()
        t = self.K + 1

        if self.n_rules_ > 0:
            acts, dists = self._compute_rule_activations_and_dists(x)

            # --- S1/S2: atividade relevante (mesma regra que nas variantes)
            a_max = float(acts.max())
            sig = np.array([r.sigma for r in self.rules])
            rel = (a_max >= self.act_min_rel) & (acts >= self.beta * a_max) \
                  & (dists <= self.dist_factor_rel * sig)
            for r, is_rel in zip(self.rules, rel):
                if is_rel:
                    if r.n_act >= 1:
                        gap = r.inactive + 1
                        self.gaps.append((t, region(r.center), gap))
                        r.gap_sum += gap
                    r.n_act += 1
                    r.inactive = 0
                else:
                    r.inactive += 1

            # --- S3: peso normalizado e uma EMA por cada alpha
            s = float(acts.sum())
            u = acts / s if s > 0 else np.zeros_like(acts)
            for r, u_r in zip(self.rules, u):
                for a in self.alphas:
                    r.ema[a] = (1 - a) * r.ema[a] + a * float(u_r)

        super()._update_one(x, y)                    # B0 aprende (inalterado)

        for r in self.rules:                         # regras novas
            if not hasattr(r, "born_t"):
                r.born_t, r.inactive, r.n_act, r.gap_sum = t, 0, 0, 0
                means = {a: np.mean([q.ema[a] for q in self.rules if hasattr(q, "ema")])
                         if any(hasattr(q, "ema") for q in self.rules) else 1.0
                         for a in self.alphas}
                r.ema = dict(means)

        if t % self.snapshot_every == 0:
            self._snapshot(t)

    def _snapshot(self, t):
        mean_ema = {a: np.mean([r.ema[a] for r in self.rules]) for a in self.alphas}
        for r in self.rules:
            self.snapshots.append({
                "t": t, "age": t - r.born_t, "region": region(r.center),
                "inactive": r.inactive, "n_act": r.n_act,
                **{f"ratio_{a}": r.ema[a] / mean_ema[a] for a in self.alphas},
            })


# ---------------------------------------------------------------------------
# 3. Resumo
# ---------------------------------------------------------------------------

def pct(v, qs=(10, 50, 90, 99)):
    v = np.asarray(v, dtype=float)
    return "  ".join(f"p{q}={np.percentile(v, q):.1f}" for q in qs) if v.size else "(vazio)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--G", type=int, default=100, help="idade mínima p/ contar no ratio")
    args = ap.parse_args()

    X, y, (b1, b2) = aba_stream(args.n, args.seed)
    m = B0Diagnostic(n_features=2, n_classes=2, max_rules=1000)
    for i in range(args.n):
        m.partial_fit(X[i:i + 1], y[i:i + 1])

    print(f"Stream A-B-A  n={args.n}  seed={args.seed}  drifts em t={b1} e t={b2}")
    print(f"Regras finais: {m.n_rules_}  (A={sum(region(r.center)=='A' for r in m.rules)}, "
          f"B={sum(region(r.center)=='B' for r in m.rules)}, "
          f"AB={sum(region(r.center)=='AB' for r in m.rules)})")
    print(f"Distância entre drifts: {b2 - b1} observações\n")

    g = [gap for _, _, gap in m.gaps]
    print("=== S1/S2: intervalos entre ativações relevantes (gaps) ===")
    print("todos           ", pct(g))
    mean_gaps = np.array([r.gap_sum / (r.n_act - 1) for r in m.rules if r.n_act >= 3])
    print("média por regra ", pct(mean_gaps), f"  ({mean_gaps.size} regras com >= 3 ativações)")
    print("\nFração de regras em que k * mean_gap > H0 (o termo adaptativo do S2 atua):")
    for H0 in (10, 20, 30, 50, 100):
        print(f"  H0={H0:<4}" + "  ".join(f"k={k}: {np.mean(k * mean_gaps > H0):.0%}" for k in (3, 5, 10)))

    snaps = m.snapshots
    in_B = [s for s in snaps if b1 < s["t"] <= b2]
    print("\n=== S1/S2: inatividade das regras durante a fase B ===")
    print("regras só-A (obsoletas)", pct([s["inactive"] for s in in_B if s["region"] == "A"]))
    print("outras regras          ", pct([s["inactive"] for s in in_B if s["region"] != "A"]))

    print(f"\n=== S3: ema / média(ema), regras com idade >= {args.G}, fase B ===")
    print("(regras só-A estão obsoletas nesta fase; as outras estão em uso)")
    for a in m.alphas:
        old = [s[f"ratio_{a}"] for s in in_B if s["region"] == "A" and s["age"] >= args.G]
        cur = [s[f"ratio_{a}"] for s in in_B if s["region"] != "A" and s["age"] >= args.G]
        q = (5, 10, 30, 50)
        print(f"alpha={a:<6} 1/alpha={1/a:>5.0f} | só-A  " +
              "  ".join(f"p{k}={np.percentile(old, k):.2f}" for k in q) +
              " | outras " + "  ".join(f"p{k}={np.percentile(cur, k):.2f}" for k in q))


if __name__ == "__main__":
    main()
