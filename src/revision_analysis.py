from __future__ import annotations
import json, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .calibration import IsotonicCalibrator, TemperatureScaling
from .selective_risk import certify, crc_style, sgr_certify, evaluate

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
DELTA = 0.10
DATASETS = ["scifact", "fever", "hotpotqa"]
ROUTERS = ["feature", "selfrag", "case", "fixed"]
STUDIES = {
    "hosted":     (ROOT/"runs"/"matrix",             "{ds}_{rt}_s{sd}",       {"case": "jev"},  [1, 2, 3]),
    "open_gemma": (ROOT/"ClefServer"/"runs"/"matrix", "gemma_{ds}_{rt}_s{sd}", {"case": "clef"}, [1, 2, 3]),
    "open_qwen":  (ROOT/"ClefServer"/"runs"/"matrix", "qwen_{ds}_{rt}_s{sd}",  {"case": "clef"}, [1, 2, 3]),
}
RNG = np.random.default_rng(20261009)


def _xy(recs):
    return (np.array([r["probabilities"]["ANSWER"] for r in recs], float),
            np.array([1.0 if r["correct"] else 0.0 for r in recs], float))


def load(study, ds, rt, sd):
    d, pat, names, _ = STUDIES[study]
    name = pat.format(ds=ds, rt=names.get(rt, rt), sd=sd)
    p = d / f"{name}.json"
    if not p.exists():
        return None
    run = json.loads(p.read_text())
    test = json.loads((d / f"{name}.records.json").read_text())
    return run["calib_records"], test


def old_rule(sc, yc):
    t = TemperatureScaling().fit(sc, yc); i = IsotonicCalibrator().fit(sc, yc)
    cal = i if (i.report and t.report and i.report.ece <= t.report.ece) else t
    return cal


def mean(v):
    return float(np.mean(v)) if len(v) else None


def certification(out):
    sweep, invalid, counts = {}, {}, Counter()
    mono = {"pairs": 0, "monotone": 0, "exceptions": []}
    for study, (_, _, _, seeds) in STUDIES.items():
        for ds, rt in itertools.product(DATASETS, ROUTERS):
            runs = [load(study, ds, rt, sd) for sd in seeds]
            runs = [(sd, r) for sd, r in zip(seeds, runs) if r is not None]
            if not runs:
                continue
            key = f"{study}|{ds}|{rt}"
            row = []
            covs_by_seed = defaultdict(list)
            for al in ALPHAS:
                cov, rk, na, nc, per = [], [], 0, 0, []
                for sd, (cal, test) in runs:
                    sc, yc = _xy(cal); st, yt = _xy(test)
                    c = sgr_certify(sc, yc, al, DELTA)
                    op = evaluate(c[0] if c else None, st, yt)
                    covs_by_seed[sd].append(op.coverage)
                    per.append({"seed": sd, "cert": c is not None, "coverage": op.coverage,
                                "risk": op.risk, "n": op.n_answered})
                    if c is not None:
                        nc += 1; cov.append(op.coverage); na += op.n_answered
                        if op.risk is not None:
                            rk.append(op.risk)
                        counts["valid_certs"] += 1
                        counts["valid_viol"] += int(op.risk is not None and op.risk > al)
                    k = crc_style(sc, yc, al)
                    if k is not None:
                        o2 = evaluate(k[0], st, yt)
                        counts["crc_certs"] += 1
                        counts["crc_viol"] += int(o2.risk is not None and o2.risk > al)
                row.append({"alpha": al, "n_certifying": nc, "n_seeds": len(runs),
                            "coverage": mean(cov), "risk": mean(rk), "n_answered": na,
                            "per_seed": per})
            sweep[key] = row
            for sd, v in covs_by_seed.items():
                mono["pairs"] += 1
                if all(b >= a - 1e-12 for a, b in zip(v, v[1:])):
                    mono["monotone"] += 1
                else:
                    mono["exceptions"].append(f"{key}|s{sd}")

            inv = {"crc": {"n_cert": 0, "n": 0, "err": 0, "risks": []},
                   "sgr": {"n_cert": 0, "n": 0, "err": 0, "risks": []}}
            for sd, (cal, test) in runs:
                sc, yc = _xy(cal); st, yt = _xy(test)
                for name, c in (("crc", crc_style(sc, yc, 0.15)), ("sgr", sgr_certify(sc, yc, 0.15, DELTA))):
                    if c is None:
                        continue
                    op = evaluate(c[0], st, yt)
                    inv[name]["n_cert"] += 1; inv[name]["n"] += op.n_answered
                    if op.risk is not None:
                        inv[name]["risks"].append(op.risk)
                        inv[name]["err"] += int(round(op.risk * op.n_answered))
            invalid[key] = inv
    out["sweep"] = sweep
    out["invalid_vs_valid"] = invalid
    out["certificate_counts"] = dict(counts)
    out["monotonicity"] = mono


    tr = {}
    for study, (_, _, _, seeds) in STUDIES.items():
        rows = []
        for sd in seeds:
            a = load(study, "scifact", "case", sd); b = load(study, "fever", "case", sd)
            if a is None or b is None:
                continue
            sc, yc = _xy(a[0]); ft, fy = _xy(b[1])
            c = sgr_certify(sc, yc, 0.15, DELTA)
            op = evaluate(c[0] if c else None, ft, fy)
            rows.append({"seed": sd, "cert": c is not None, "coverage": op.coverage, "risk": op.risk})
        tr[study] = rows
    out["transfer_scifact_to_fever"] = tr


def simulation(out, reps=2000):
    res = {}
    for study, (_, _, _, seeds) in STUDIES.items():
        for ds in DATASETS:
            r = load(study, ds, "case", seeds[0])
            if r is None:
                continue
            s1, y1 = _xy(r[0]); s2, y2 = _xy(r[1])
            S, Y = np.concatenate([s1, s2]), np.concatenate([y1, y2])
            for al in (0.15, 0.20, 0.30):
                tally = Counter()
                for _ in range(reps):
                    idx = RNG.integers(0, len(S), 300)
                    sc, yc = S[idx], Y[idx]
                    rules = {"sgr": sgr_certify(sc, yc, al, DELTA), "crc": crc_style(sc, yc, al)}
                    cal = old_rule(sc, yc)
                    oc = certify(cal.predict_raw(sc), yc, al, DELTA)
                    if oc is not None:
                        sel = cal.predict_raw(S) >= oc[0]
                        rules["old"] = ("mask", sel)
                    else:
                        rules["old"] = None
                    for name, c in rules.items():
                        if c is None:
                            continue
                        sel = c[1] if c[0] == "mask" else S >= c[0]
                        if sel.sum() == 0:
                            continue
                        tally[name + "_cert"] += 1
                        tally[name + "_viol"] += int((1 - Y[sel]).mean() > al)
                res[f"{study}|{ds}|{al}"] = {k: v / reps for k, v in tally.items()}
    out["simulation"] = {"reps": reps, "delta": DELTA, "rows": res}


def cluster_boot(diffs_by_id, B=10000):
    ids = list(diffs_by_id)
    sums = np.array([sum(diffs_by_id[i]) for i in ids], float)
    cnts = np.array([len(diffs_by_id[i]) for i in ids], float)
    point = sums.sum() / cnts.sum()
    idx = RNG.integers(0, len(ids), size=(B, len(ids)))
    samp = sums[idx].sum(1) / cnts[idx].sum(1)
    return 100 * point, 100 * np.percentile(samp, 2.5), 100 * np.percentile(samp, 97.5), len(ids), int(cnts.sum())


def accuracy_intervals(out):
    res = {}
    for study, (_, _, _, seeds) in STUDIES.items():
        for ds in DATASETS:
            for base in ("fixed", "feature", "selfrag"):
                d = defaultdict(list); per_seed = []
                for sd in seeds:
                    a = load(study, ds, "case", sd); b = load(study, ds, base, sd)
                    if a is None or b is None:
                        continue
                    A = {r["id"]: r for r in a[1]}; Bm = {r["id"]: r for r in b[1]}
                    sh = sorted(set(A) & set(Bm))
                    dd = defaultdict(list)
                    for i in sh:
                        v = float(Bm[i]["correct"]) - float(A[i]["correct"])
                        d[i].append(v); dd[i].append(v)
                    per_seed.append((sd,) + cluster_boot(dd)[:3])
                if d:
                    p, lo, hi, nid, nobs = cluster_boot(d)
                    res[f"{study}|{ds}|{base}"] = {"delta_base_minus_case_pts": p, "ci": [lo, hi],
                                                   "n_ids": nid, "n_obs": nobs, "per_seed": per_seed}
    out["accuracy_cluster_bootstrap"] = res


def actions(out):
    res = {}
    for study, (_, _, _, seeds) in STUDIES.items():
        for ds in DATASETS:
            for sd in seeds:
                r = load(study, ds, "case", sd)
                if r is None:
                    continue
                for split, recs in (("calib", r[0]), ("test", r[1])):
                    c = Counter(x["action"] for x in recs)
                    multi = sum(1 for x in recs if x["n_retrievals"] > 1)
                    full = sum(1 for x in recs if x["n_retrievals"] >= 3)
                    res[f"{study}|{ds}|s{sd}|{split}"] = {"terminal": dict(c), "n": len(recs),
                                                          "second_round": multi, "third_round": full}
    out["actions"] = res


def main():
    out = {"alphas": ALPHAS, "delta": DELTA, "rule": "SGR on raw ANSWER score"}
    certification(out)
    accuracy_intervals(out)
    actions(out)
    simulation(out)
    p = ROOT / "results" / "revision.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
