"""T09 baseline demonstration: UnimNeuron Safe, no pruning modifications.

Place this file in the root of the cloned UnimNeuron-EvolvingNF repository.
Requirements: numpy (and ENF_UnimNeuron_Safe.py from the repository).

Usage: python t09_baseline_demo.py --n 900 --warmup 50 --max-rules 50
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe


def make_stream(n: int, seed: int):
    """In-memory, chronological synthetic stream with known recurring drift.

    A: y = 1[x0 > 0.5]; B: y = 1[x1 > 0.5]; A returns.
    x0,x1 are generated in [0,1]; no preprocessing fitted on future data.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 2))
    boundaries = (n // 3, 2 * n // 3)
    y = np.empty(n, dtype=int)
    phases = []
    for i in range(n):
        phase = "A" if i < boundaries[0] or i >= boundaries[1] else "B"
        phases.append(phase)
        y[i] = int(X[i, 0] > 0.5) if phase == "A" else int(X[i, 1] > 0.5)
    return X, y, phases, boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-rules", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", type=int, default=12, help="Number of first samples to print")
    parser.add_argument("--output", default="t09_smoke_trace.csv")
    args = parser.parse_args()
    if args.n < 6 or not 0 <= args.warmup < args.n or args.max_rules < 1:
        parser.error("Require n>=6, 0<=warmup<n, and max-rules>=1")

    X, y, phase, boundaries = make_stream(args.n, args.seed)
    model = EvolvingNeuroFuzzyUnimSafe(
        n_features=2, n_classes=2, max_rules=args.max_rules, random_state=args.seed
    )

    print(f"Stream: {args.n} amostras; A -> B na observacao {boundaries[0]+1}; "
          f"B -> A na {boundaries[1]+1}")
    print(f"Baseline ORIGINAL | max_rules={args.max_rules} | warm-up={args.warmup}")
    print("Primeiras observacoes: prevemos ANTES de aprender com o label atual")
    print("  t | fase |   x0    x1 | y_real y_prev | regras antes -> depois")

    rows = []
    for t, (x_t, y_t, p_t) in enumerate(zip(X, y, phase), start=1):
        before = model.n_rules_
        # At t=1 model is empty: documented fallback class 0.
        y_hat = int(model.predict(x_t.reshape(1, -1))[0])
        correct = int(y_hat == y_t)
        model.partial_fit(x_t.reshape(1, -1), np.array([y_t]))
        after = model.n_rules_
        rows.append((t, p_t, float(x_t[0]), float(x_t[1]), int(y_t), y_hat, correct, before, after))
        if t <= args.show:
            print(f"{t:>3} | {p_t:>4} | {x_t[0]:.3f} {x_t[1]:.3f} | "
                  f"{y_t:^6} {y_hat:^6} | {before:>3} -> {after:<3}")

    print("\nResumo prequential (accuracy calculada EXCLUINDO warm-up):")
    eligible = rows[args.warmup:]
    overall = sum(r[6] for r in eligible) / len(eligible)
    print(f"  Accuracy pos warm-up: {overall:.3f} ({sum(r[6] for r in eligible)}/{len(eligible)})")
    for name in ("A1", "B", "A2"):
        if name == "A1": subset = [r for r in eligible if r[0] <= boundaries[0]]
        elif name == "B": subset = [r for r in eligible if boundaries[0] < r[0] <= boundaries[1]]
        else: subset = [r for r in eligible if r[0] > boundaries[1]]
        if subset:
            print(f"  {name:>2}: accuracy={sum(r[6] for r in subset)/len(subset):.3f}; "
                  f"amostras={len(subset)}")
    print(f"  Regras finais: {model.n_rules_}; "
          f"regras medias pos warm-up: {np.mean([r[8] for r in eligible]):.2f}")
    print("  Aviso: ainda NAO existe a variante T09 nem foram reproduzidos os benchmarks oficiais.")

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "x0", "x1", "y_true", "y_pred", "correct", "rules_before", "rules_after"])
        w.writerows(rows)
    print(f"  Trace guardado em: {out.resolve()}")


if __name__ == "__main__":
    main()
