from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class CalibrationReport:
    method: str
    n: int
    ece: float
    brier: float
    nll: float


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               bins: int = 15) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def brier_score(conf: np.ndarray, correct: np.ndarray) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    return float(np.mean((conf - correct) ** 2))


def neg_log_loss(conf: np.ndarray, correct: np.ndarray) -> float:
    conf = np.clip(np.asarray(conf, dtype=float), 1e-6, 1 - 1e-6)
    correct = np.asarray(correct, dtype=float)
    return float(-np.mean(correct * np.log(conf) +
                          (1 - correct) * np.log(1 - conf)))


class TemperatureScaling:
    def __init__(self, n_grid: int = 200) -> None:
        self.temperature = 1.0
        self.report: CalibrationReport | None = None

    def fit(self, conf: Sequence[float], correct: Sequence[float]) -> "TemperatureScaling":
        conf = np.clip(np.asarray(conf, dtype=float), 1e-6, 1 - 1e-6)
        correct = np.asarray(correct, dtype=float)
        odds = np.log(conf / (1 - conf))
        best_t, best_nll = 1.0, math.inf
        for t in np.linspace(0.05, 5.0, 200):
            p = 1 / (1 + np.exp(-(odds / t)))
            nll = -np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p))
            if nll < best_nll:
                best_t, best_nll = t, float(nll)
        self.temperature = best_t
        self.report = CalibrationReport("temperature", len(conf),
                                        expected_calibration_error(self.predict_raw(conf), correct),
                                        brier_score(self.predict_raw(conf), correct),
                                        best_nll)
        return self

    def predict_raw(self, conf: np.ndarray) -> np.ndarray:
        conf = np.clip(np.asarray(conf, dtype=float), 1e-6, 1 - 1e-6)
        odds = np.log(conf / (1 - conf))
        return 1 / (1 + np.exp(-(odds / self.temperature)))

    def __call__(self, score: float) -> float:
        return float(self.predict_raw(np.array([score]))[0])


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.report: CalibrationReport | None = None

    def fit(self, conf: Sequence[float], correct: Sequence[float]) -> "IsotonicCalibrator":
        conf = np.asarray(conf, dtype=float)
        correct = np.asarray(correct, dtype=float)
        order = np.argsort(conf)
        x, y = conf[order], correct[order]

        blocks = [(float(y[i]), 1, float(x[i])) for i in range(len(y))]
        merged: list[tuple[float, int, float]] = []
        for val, w, xv in blocks:
            merged.append([val, w, xv])
            while len(merged) >= 2 and merged[-2][0] > merged[-1][0]:
                v2, w2, x2 = merged.pop()
                v1, w1, x1 = merged.pop()
                total = w1 + w2
                merged.append([(v1 * w1 + v2 * w2) / total, total, x1])
        xs, ys = [], []
        for val, w, xv in merged:
            xs.append(xv)
            ys.append(val)
        self.x_, self.y_ = np.array(xs), np.array(ys)
        pred = self.predict_raw(conf)
        self.report = CalibrationReport(
            "isotonic", len(conf), expected_calibration_error(pred, correct),
            brier_score(pred, correct), neg_log_loss(pred, correct))
        return self

    def predict_raw(self, conf: np.ndarray) -> np.ndarray:
        assert self.x_ is not None and self.y_ is not None
        conf = np.asarray(conf, dtype=float)
        return np.interp(conf, self.x_, self.y_)

    def __call__(self, score: float) -> float:
        return float(self.predict_raw(np.array([score]))[0])


class ConformalRiskControl:
    def __init__(self, target_risk: float = 0.10) -> None:
        self.target_risk = target_risk
        self.threshold: float | None = None
        self.calib_risk: float | None = None
        self.calib_coverage: float | None = None

    def fit(self, conf: Sequence[float], correct: Sequence[float]) -> "ConformalRiskControl":
        conf = np.asarray(conf, dtype=float)
        correct = np.asarray(correct, dtype=float)
        n = len(conf)
        candidates = np.sort(np.unique(conf))[::-1]
        best = None
        for t in candidates:
            answered = conf >= t
            if answered.sum() == 0:
                continue
            risk = float(1.0 - correct[answered].mean())
            coverage = float(answered.mean())


            if risk * (n / (n + 1)) + 1 / (n + 1) <= self.target_risk:
                best = (t, risk, coverage)
            else:
                break
        if best is None:
            self.threshold = None
            self.calib_risk = None
            self.calib_coverage = 0.0
        else:
            self.threshold = float(best[0])
            self.calib_risk = float(best[1])
            self.calib_coverage = float(best[2])
        return self

    @property
    def certifiable(self) -> bool:
        return self.threshold is not None

    def should_answer(self, score: float) -> bool:
        if self.threshold is None:
            return False
        return score >= self.threshold


def fit_calibrator(method: str, conf: Sequence[float], correct: Sequence[float]):
    if method == "temperature":
        return TemperatureScaling().fit(conf, correct)
    if method == "isotonic":
        return IsotonicCalibrator().fit(conf, correct)
    if method == "none":
        return None
    raise ValueError(f"unknown calibrator: {method}")
