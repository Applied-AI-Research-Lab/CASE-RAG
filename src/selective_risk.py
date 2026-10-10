from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

DEFAULT_DELTA = 0.10


def certify(conf, correct, alpha: float, delta: float = DEFAULT_DELTA):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    best = None
    for t in np.sort(np.unique(conf))[::-1]:
        answered = conf >= t
        m = int(answered.sum())
        if m == 0:
            continue
        k = int((1.0 - correct[answered]).sum())
        if binom.cdf(k, m, alpha) <= delta:
            best = (float(t), m, k)
        else:
            break
    return best


def sgr_certify(conf, correct, alpha: float, delta: float = DEFAULT_DELTA):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    z = np.sort(conf)
    n = len(z)
    steps = int(np.ceil(np.log2(n)))
    d = delta / steps
    lo, hi = 0, n - 1
    best = None
    for _ in range(steps):
        mid = int(np.ceil((lo + hi) / 2))
        t = z[mid]
        answered = conf >= t
        m = int(answered.sum())
        k = int((1.0 - correct[answered]).sum())
        if m > 0 and binom.cdf(k, m, alpha) <= d:
            best = (float(t), m, k)
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1:
            break
    return best


def crc_style(conf, correct, alpha: float):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    best = None
    for t in np.sort(np.unique(conf))[::-1]:
        answered = conf >= t
        m = int(answered.sum())
        if m == 0:
            continue
        r = float(1.0 - correct[answered].mean())
        if r * (n / (n + 1)) + 1 / (n + 1) <= alpha:
            best = (float(t), m, int(round(r * m)))
        else:
            break
    return best


@dataclass
class Operating:
    coverage: float
    risk: float | None
    n_answered: int


def evaluate(threshold, conf, correct) -> Operating:
    if threshold is None:
        return Operating(0.0, None, 0)
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    a = conf >= threshold
    m = int(a.sum())
    if m == 0:
        return Operating(0.0, None, 0)
    return Operating(m / len(conf), float((1.0 - correct[a]).mean()), m)
