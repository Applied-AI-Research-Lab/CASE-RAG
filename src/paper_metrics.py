from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from .calibration import IsotonicCalibrator, TemperatureScaling, expected_calibration_error
from .evaluation.metrics import aurc_tie_aware

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["scifact", "fever", "hotpotqa"]
STUDIES = {
    "hosted":     (ROOT/"runs"/"matrix",            "{ds}_{rt}_s{sd}",       {"case": "jev"},  [1, 2, 3]),
    "open_gemma": (ROOT/"ClefServer"/"runs"/"matrix","gemma_{ds}_{rt}_s{sd}", {"case": "clef"}, [1, 2, 3]),
    "open_qwen":  (ROOT/"ClefServer"/"runs"/"matrix","qwen_{ds}_{rt}_s{sd}",  {"case": "clef"}, [1, 2, 3]),
    "glm":        (ROOT/"runs"/"matrix_glm",        "{ds}_{rt}_s{sd}",       {"case": "jev"},  [1, 2]),
}
ROUTERS = ["feature", "selfrag", "case", "fixed"]


def _xy(recs):
    return (np.array([r["probabilities"]["ANSWER"] for r in recs], float),
            np.array([1.0 if r["correct"] else 0.0 for r in recs], float))


def _recal(sc, yc):
    t = TemperatureScaling().fit(sc, yc); i = IsotonicCalibrator().fit(sc, yc)
    return i if (i.report and t.report and i.report.ece <= t.report.ece) else t


def _runs(study, ds, rt):
    d, pat, names, seeds = STUDIES[study]
    for sd in seeds:
        name = pat.format(ds=ds, rt=names.get(rt, rt), sd=sd)
        p, q = d / f"{name}.json", d / f"{name}.records.json"
        if p.exists() and q.exists():
            yield sd, json.loads(p.read_text())["calib_records"], json.loads(q.read_text())


def _lr(x_recs):
    return np.array([[r["evidence_sufficiency"], r["n_retrievals"], r["n_evidence_passages"],
                      np.log1p(r["n_evidence_chars"])] for r in x_recs], float)


def main():
    out = {}
    for study in STUDIES:
        for ds in DATASETS:
            for rt in ROUTERS + (["regression"] if study.startswith("open") else []):
                src = "case" if rt == "regression" else rt
                per = []
                for sd, cal, test in _runs(study, ds, src):
                    sc, yc = _xy(cal); st, yt = _xy(test)
                    if rt == "regression":
                        lr = LogisticRegression(max_iter=2000).fit(_lr(cal), yc)
                        sc, st = lr.predict_proba(_lr(cal))[:, 1], lr.predict_proba(_lr(test))[:, 1]
                    g = _recal(sc, yc)
                    per.append({"seed": sd, "acc": float(yt.mean()),
                                "ece": expected_calibration_error(g.predict_raw(st), yt),
                                "raw_ece": expected_calibration_error(st, yt),
                                "aurc": aurc_tie_aware(st, yt),
                                "retr": float(np.mean([r["n_retrievals"] for r in test])),
                                "distinct": int(len(np.unique(st))),
                                "rails": float(((st < 0.05) | (st > 0.95)).mean())})
                if not per:
                    continue
                agg = {k: float(np.mean([p[k] for p in per])) for k in per[0] if k != "seed"}
                agg.update({k + "_sd": float(np.std([p[k] for p in per])) for k in ("acc", "ece", "aurc", "retr")})
                agg["n_seeds"] = len(per); agg["per_seed"] = per
                out[f"{study}|{ds}|{rt}"] = agg
    p = ROOT / "results" / "paper_metrics.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
