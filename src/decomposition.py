from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "matrix"
DATASETS = ["scifact", "fever", "hotpotqa"]
SEEDS = [1, 2, 3]
LABEL = {"feature": "Heuristic", "selfrag": "Self-RAG", "jev": "CASE-RAG"}


def recs(ds, rt, seed):
    p = RUNS / f"{ds}_{rt}_s{seed}.records.json"
    return {r["id"]: r for r in json.loads(p.read_text())} if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/decomposition.json")
    args = ap.parse_args()
    rows = []
    print(f"{'dataset':9s} {'baseline':9s} | {'agree n':>7s} {'agree dacc':>10s} | "
          f"{'differ n':>8s} {'differ dacc':>11s} | {'ret base':>8s} {'ret case':>8s}")
    for ds in DATASETS:
        for base in ("feature", "selfrag"):
            ag_n = ag_d = df_n = df_d = 0
            rb, rj = [], []
            for sd in SEEDS:
                A, B = recs(ds, base, sd), recs(ds, "jev", sd)
                if not A or not B:
                    continue
                for i in sorted(set(A) & set(B)):
                    ca, cb = float(A[i]["correct"]), float(B[i]["correct"])
                    rb.append(A[i]["n_retrievals"])
                    rj.append(B[i]["n_retrievals"])
                    if A[i]["n_retrievals"] == B[i]["n_retrievals"]:
                        ag_n += 1
                        ag_d += cb - ca
                    else:
                        df_n += 1
                        df_d += cb - ca
            if not (ag_n + df_n):
                continue
            print(f"{ds:9s} {LABEL[base]:9s} | {ag_n:7d} {100*ag_d/max(ag_n,1):+10.2f} | "
                  f"{df_n:8d} {100*df_d/max(df_n,1):+11.2f} | "
                  f"{np.mean(rb):8.2f} {np.mean(rj):8.2f}")
            rows.append({"dataset": ds, "baseline": base,
                         "n_agree": ag_n, "delta_acc_agree_pts": 100 * ag_d / max(ag_n, 1),
                         "n_differ": df_n, "delta_acc_differ_pts": 100 * df_d / max(df_n, 1),
                         "share_of_total_delta_from_differ": 1.0 if ag_d == 0 else None,
                         "mean_retrievals_baseline": float(np.mean(rb)),
                         "mean_retrievals_case": float(np.mean(rj))})
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
