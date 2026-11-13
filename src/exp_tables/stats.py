"""The statistics, in the standard library: a summary over seeds (mean, standard deviation, standard error, a
confidence interval) and Student's t distribution, computed from the regularized incomplete beta function."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b).

    >>> round(betainc(2, 3, 0.4), 6)  # 1 - (1-x)^4 - 4x(1-x)^3 ... = 0.5248
    0.5248
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom.

    >>> round(t_cdf(2.228, 10), 4), round(t_cdf(0, 5), 4), round(t_cdf(-1.96, 1e9), 3)
    (0.975, 0.5, 0.025)
    """
    if df > 1e7:  # the normal distribution
        return 0.5 * math.erfc(-t / math.sqrt(2))
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


@lru_cache(maxsize=4096)
def t_ppf(p: float, df: float) -> float:
    """The t such that P(T <= t) = p: bisection on t_cdf, in a bracket that doubles until it holds the answer; the
    normal quantile past a million degrees of freedom. Cached: power calculations ask for the same few again and again.

    >>> round(t_ppf(0.975, 10), 3), round(t_ppf(0.975, 1), 3), round(t_ppf(0.975, 1e9), 3)
    (2.228, 12.706, 1.96)
    >>> round(t_ppf(0.9995, 1), 1)  # beyond the old bracket's reach in spirit: 636.6
    636.6
    """
    if df > 1e6:
        return NormalDist().inv_cdf(p)
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1 - p, df)
    lo, hi = 0.0, 2.0
    while t_cdf(hi, df) < p and hi < 1e12:
        lo, hi = hi, hi * 2
    while hi - lo > 1e-12 * max(1.0, hi):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Summary:
    """The values of one metric over seeds."""

    values: list[float]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        if self.values and min(self.values) == max(self.values):
            return self.values[0]  # exactly, not 0.1 + 0.1 + 0.1 over 3
        return math.fsum(self.values) / self.n

    @property
    def std(self) -> float:
        """The sample standard deviation (n - 1); 0 for a single value, and exactly 0 for equal values.

        >>> Summary([0.1, 0.1, 0.1]).std
        0.0
        """
        if self.n < 2 or min(self.values) == max(self.values):
            return 0.0
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / (self.n - 1))

    @property
    def sem(self) -> float:
        return self.std / math.sqrt(self.n) if self.n else 0.0

    def ci(self, level: float = 0.95) -> float:
        """The half-width of the t confidence interval of the mean; 0 for a single value.

        >>> round(Summary([93.0, 93.4, 92.9, 93.3, 93.1]).ci(), 3)  # t(0.975, 4) = 2.776, times 0.2074 / sqrt(5)
        0.257
        """
        if self.n < 2:
            return 0.0
        return t_ppf(0.5 + level / 2, self.n - 1) * self.sem
