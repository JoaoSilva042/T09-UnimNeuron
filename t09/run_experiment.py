"""Run one prequential experiment (test-then-train) and save CSV + JSON.

Usage (from the repository root):
    uv run python -m t09.run_experiment --stream aba --variant S1 --seed 100 --n 5000 --param H=100
    uv run python -m t09.run_experiment --stream aba --variant S3 --seed 100 --param alpha=0.01 --param theta=0.3

Output:
    results/<stream>/<variant>[_<params>]_seed<k>.csv   one line per observation
    results/<stream>/<variant>[_<params>]_seed<k>.json  configuration + summary
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from t09.prunning import B0Counted, S1Fixed, S2Adaptive, S3EMA
from t09.streams import STREAMS

VARIANTS = {"B0": B0Counted, "S1": S1Fixed, "S2": S2Adaptive, "S3": S3EMA}
CSV_HEADER = ["t", "y_true", "y_pred", "n_rules", "pruned_now", "created_now",
              "recreated_now", "capacity_deletions", "time_s"]


def parse_params(items):
    """['H=100', 'theta=0.3'] -> {'H': 100, 'theta': 0.3}"""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            v = float(v)
        out[k] = v
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def output_name(stream, variant, seed, params, out_dir="results"):
    tag = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    name = f"{variant}{'_' + tag if tag else ''}_seed{seed}"
    return Path(out_dir) / stream, name


def run(stream, variant, seed, n, max_rules=1000, params=None, warmup=0, out_dir="results",
        overwrite=False):
    params = params or {}
    folder, name = output_name(stream, variant, seed, params, out_dir)
    if (folder / f"{name}.csv").exists() and not overwrite:
        raise FileExistsError(f"{folder / name}.csv already exists (use --overwrite to replace it)")
    X, y, info = STREAMS[stream](n, seed)
    model = VARIANTS[variant](n_features=info["n_features"], n_classes=info["n_classes"],
                              max_rules=max_rules, **params)

    rows = []
    t_start = time.perf_counter()
    for i in range(n):
        x_i, y_i = X[i:i + 1], y[i:i + 1]
        before = (model.n_pruned_, model.n_created, model.n_recreated, model.capacity_deletions)
        t0 = time.perf_counter()
        y_hat = int(model.predict(x_i)[0])       # 1. test ...
        model.partial_fit(x_i, y_i)              # 2. ... then train
        dt = time.perf_counter() - t0
        rows.append((i + 1, int(y_i[0]), y_hat, model.n_rules_,
                     model.n_pruned_ - before[0], model.n_created - before[1],
                     model.n_recreated - before[2], model.capacity_deletions - before[3],
                     round(dt, 6)))
    total = time.perf_counter() - t_start

    # ---- files
    folder.mkdir(parents=True, exist_ok=True)

    with (folder / f"{name}.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    arr = np.array([r[:4] for r in rows])
    eval_part = arr[warmup:]
    summary = {
        "accuracy_after_warmup": float(np.mean(eval_part[:, 1] == eval_part[:, 2])),
        "rules_mean": float(np.mean(arr[:, 3])), "rules_max": int(np.max(arr[:, 3])),
        "rules_final": int(arr[-1, 3]),
        "n_created": model.n_created, "n_pruned": model.n_pruned_,
        "n_recreated": model.n_recreated, "n_class_blocks": model.n_class_blocks,
        "capacity_deletions": model.capacity_deletions,
        "valid": model.capacity_deletions == 0,
        "time_total_s": round(total, 3), "time_per_obs_ms": round(1000 * total / n, 4),
    }
    config = {
        "stream": stream, "stream_info": info, "seed": seed, "n": n,
        "variant": variant, "model_class": type(model).__name__,
        "max_rules": max_rules, "params": params,
        "params_effective": {k: getattr(model, k) for k in
                             ("beta", "act_min_rel", "dist_factor_rel", "H", "prune_every",
                              "recreate_window", "k", "n_min", "alpha", "theta", "G")
                             if hasattr(model, k)},
        "warmup_used_in_summary": warmup,
        "summary": summary,
        "git_commit": git_commit(), "date": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "numpy": np.__version__,
    }
    with (folder / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return folder / f"{name}.csv", summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=sorted(STREAMS))
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--max-rules", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=0,
                    help="only for the quick summary in the .json; the CSV keeps every observation")
    ap.add_argument("--param", action="append", help="hyperparameter, e.g. --param H=100")
    ap.add_argument("--out", default="results")
    ap.add_argument("--overwrite", action="store_true", help="replace existing results")
    a = ap.parse_args()

    path, s = run(a.stream, a.variant, a.seed, a.n, a.max_rules, parse_params(a.param), a.warmup, a.out, a.overwrite)
    print(f"{path}  acc={s['accuracy_after_warmup']:.4f}  rules_mean={s['rules_mean']:.1f}  "
          f"pruned={s['n_pruned']}  recreated={s['n_recreated']}  valid={s['valid']}  "
          f"{s['time_per_obs_ms']} ms/obs")
    if not s["valid"]:
        print("ATENÇÃO: houve remoções por capacidade (max_rules) -> corrida inválida.")


if __name__ == "__main__":
    main()
