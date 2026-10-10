from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from .calibration import IsotonicCalibrator, TemperatureScaling, expected_calibration_error
from .evaluation import area_over_risk_coverage
from .selective_risk import sgr_certify, evaluate

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "ClefServer" / "runs" / "matrix"
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FEATURES = ("verifier support", "retrieval rounds", "evidence passages", "log evidence chars")

def _x(recs):
    return np.array([[r["evidence_sufficiency"], r["n_retrievals"], r["n_evidence_passages"],
                      np.log1p(r["n_evidence_chars"])] for r in recs], float)
def _y(recs):
    return np.array([1.0 if r["correct"] else 0.0 for r in recs], float)
def _recal(sc, yc, st):
    t = TemperatureScaling().fit(sc, yc); i = IsotonicCalibrator().fit(sc, yc)
    c = i if (i.report and t.report and i.report.ece <= t.report.ece) else t
    return c.predict_raw(sc), c.predict_raw(st)

def main():
    out = {"features": list(FEATURES), "alphas": ALPHAS, "rows": {}}
    for bb in ("gemma", "qwen"):
        for ds in ("scifact", "fever", "hotpotqa"):
            acc = {n: {"aurc": [], "ece": [], **{str(a): {"cov": [], "risk": [], "n": 0} for a in ALPHAS}}
                   for n in ("lr", "case")}
            for sd in "123":
                p = MATRIX / f"{bb}_{ds}_clef_s{sd}.json"
                run = json.loads(p.read_text())
                test = json.loads((MATRIX / f"{bb}_{ds}_clef_s{sd}.records.json").read_text())
                cal = run["calib_records"]; yc, yt = _y(cal), _y(test)
                lr = LogisticRegression(max_iter=2000).fit(_x(cal), yc)
                pairs = {"lr": (lr.predict_proba(_x(cal))[:, 1], lr.predict_proba(_x(test))[:, 1]),
                         "case": (np.array([r["probabilities"]["ANSWER"] for r in cal], float),
                                  np.array([r["probabilities"]["ANSWER"] for r in test], float))}
                for name, (sc, st) in pairs.items():
                    sc2, st2 = _recal(sc, yc, st)
                    acc[name]["aurc"].append(area_over_risk_coverage(st2, yt))
                    acc[name]["ece"].append(expected_calibration_error(st2, yt))
                    for a in ALPHAS:
                        c = sgr_certify(sc, yc, a)
                        if not c: continue
                        op = evaluate(c[0], st, yt); k = acc[name][str(a)]
                        k["n"] += 1; k["cov"].append(op.coverage)
                        if op.risk is not None: k["risk"].append(op.risk)
            m = lambda v: float(np.mean(v)) if len(v) else None
            out["rows"][f"{bb}|{ds}"] = {
                n: {"aurc": m(acc[n]["aurc"]), "ece": m(acc[n]["ece"]),
                    **{str(a): {"n_certifying": acc[n][str(a)]["n"], "coverage": m(acc[n][str(a)]["cov"]),
                                "risk": m(acc[n][str(a)]["risk"])} for a in ALPHAS}}
                for n in ("lr", "case")}
    (ROOT / "results" / "feature_baseline.json").write_text(json.dumps(out, indent=1))
    print("wrote results/feature_baseline.json")

if __name__ == "__main__":
    main()
