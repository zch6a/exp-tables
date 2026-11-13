"""Is one method really better than another, or within the noise of the seeds? Tests (Welch's and the paired t
test, the Wilcoxon signed-rank test, a bootstrap test), corrections for testing many pairs at once (Holm,
Benjamini-Hochberg), effect sizes (Hedges' g, Cliff's delta), and how many seeds a difference of the size seen would
need. Everything in the standard library."""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import combinations

from .stats import Summary, t_cdf, t_ppf


@dataclass
class Result:
    name: str
    statistic: float
    p: float  # two-sided
    df: float | None = None


def welch(a: list[float], b: list[float]) -> Result:
    """Welch's t test (unequal variances), two-sided.

    >>> r = welch([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])  # t = -3 / sqrt(2.5), df = 6.25 / 1.0625; p checked by integration
    >>> round(r.statistic, 4), round(r.df, 3), round(r.p, 4)
    (-1.8974, 5.882, 0.1075)
    """
    sa, sb = Summary(a), Summary(b)
    va, vb = sa.std**2 / sa.n, sb.std**2 / sb.n
    if va + vb == 0:
        return Result("Welch's t", 0.0 if sa.mean == sb.mean else math.copysign(math.inf, sa.mean - sb.mean), 1.0 if sa.mean == sb.mean else 0.0)
    t = (sa.mean - sb.mean) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / ((va**2 / (sa.n - 1) if sa.n > 1 else 0) + (vb**2 / (sb.n - 1) if sb.n > 1 else 0))
    return Result("Welch's t", t, 2 * t_cdf(-abs(t), df), df)


def paired(a: list[float], b: list[float]) -> Result:
    """The paired t test on the differences a - b (the same seeds), two-sided.

    >>> r = paired([93.0, 93.4, 92.9, 93.3], [92.1, 92.4, 92.0, 92.5])  # differences 0.9, 1.0, 0.9, 0.8: 0.9 / (0.0816 / 2)
    >>> round(r.statistic, 3), r.df, r.p < 0.001
    (22.045, 3, True)
    """
    d = Summary([x - y for x, y in zip(a, b, strict=True)])
    if d.std == 0:
        return Result("paired t", 0.0 if d.mean == 0 else math.copysign(math.inf, d.mean), 1.0 if d.mean == 0 else 0.0, d.n - 1)
    t = d.mean / d.sem
    return Result("paired t", t, 2 * t_cdf(-abs(t), d.n - 1), d.n - 1)


def _ranks(values: list[float]) -> list[float]:
    """Ranks from 1, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda k: values[k])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def wilcoxon(a: list[float], b: list[float]) -> Result:
    """The Wilcoxon signed-rank test on paired values, two-sided: exact for up to 25 pairs (tied differences too, over
    their average ranks), the normal approximation (with ties corrected) beyond. Zero differences are dropped;
    differences are compared at 12 significant digits, so that 90.1 - 80.1 and 90.2 - 80.2 tie as they should.

    >>> r = wilcoxon([2, 3, 4, 5, 6], [1, 1, 1, 1, 1])  # every difference positive: P = 2 / 2^5
    >>> r.statistic, r.p
    (0.0, 0.0625)
    >>> wilcoxon([90.1, 90.2, 90.3, 90.4, 90.5], [80.1, 80.2, 80.3, 80.4, 80.5]).p  # five tied differences: still 2 / 2^5
    0.0625
    >>> wilcoxon([1, 2, 3, 4], [0, 0, 0, 5]).p  # d = 1, 2, 3, -1: ranks 1.5, 3, 4, 1.5; W = 1.5; {}, {1.5}, {1.5'}: 2 * 3/16
    0.375
    """
    scale = max((abs(v) for v in (*a, *b)), default=0.0) or 1.0
    d = [round((x - y) / scale, 11) for x, y in zip(a, b, strict=True)]  # ties at 11 digits of the values, not of the differences
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return Result("Wilcoxon", 0.0, 1.0)
    ranks = _ranks([abs(x) for x in d])
    w_plus = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    w = min(w_plus, n * (n + 1) / 2 - w_plus)
    if n <= 25:
        # the number of sign patterns with each sum of (doubled, so whole) ranks, by dynamic programming
        doubled = [round(2 * r) for r in ranks]
        counts = [1] + [0] * sum(doubled)
        for k in doubled:
            for s in range(len(counts) - 1, k - 1, -1):
                counts[s] += counts[s - k]
        p = min(1.0, 2 * sum(counts[: round(2 * w) + 1]) / 2**n)
        return Result("Wilcoxon", w, p)
    mean = n * (n + 1) / 4
    ties: dict[float, int] = {}
    for r in ranks:
        ties[r] = ties.get(r, 0) + 1
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t**3 - t for t in ties.values()) / 48
    z = (w - mean + 0.5) / math.sqrt(var) if var > 0 else 0.0  # continuity correction (w is the smaller sum)
    return Result("Wilcoxon", w, min(1.0, 2 * t_cdf(z, 1e9)))


def bootstrap(a: list[float], b: list[float], resamples: int = 10_000, seed: int = 0, levels: list[float] | None = None) -> Result:
    """A bootstrap test of the difference in means, two-sided: both groups resampled after shifting them to a common
    mean (so the resamples are what equal means would give); P is the share at least as far from 0 as the difference
    seen. Seeded: the same answer every time. Given the levels P will be held to (after a correction, alpha / k up to
    alpha), it stops early (after 2,000 resamples, in steps of 1,000) once P is more than four standard errors from
    every one of them: no decision would change.

    >>> bootstrap([1, 2, 3], [1, 2, 3]).p, bootstrap([10.0, 10.1, 9.9, 10.05], [1.0, 1.1, 0.9, 1.05]).p
    (1.0, 0.0)
    """
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    seen = ma - mb
    pooled = (sum(a) + sum(b)) / (na + nb)
    sa = [x - ma + pooled for x in a]
    sb = [x - mb + pooled for x in b]
    far = abs(seen) - 1e-12
    extreme = done = 0
    while done < resamples:
        batch = min(1000, resamples - done)
        for _ in range(batch):
            if abs(sum(rng.choices(sa, k=na)) / na - sum(rng.choices(sb, k=nb)) / nb) >= far:
                extreme += 1
        done += batch
        if levels and done >= 2000:
            p = extreme / done
            if all(abs(p - level) > 4 * math.sqrt(max(p * (1 - p), 1 / done) / done) for level in levels):
                break
    return Result("bootstrap", seen, extreme / done)


def holm(ps: list[float]) -> list[float]:
    """Holm's step-down correction for testing several pairs at once (controls the chance of any false claim).

    >>> holm([0.01, 0.04, 0.03])
    [0.03, 0.06, 0.06]
    """
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    out, running = [0.0] * len(ps), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[k]))
        out[k] = round(running, 15)  # 3 * 0.01 is 0.03, but 0.04999999999999 stays below 0.05
    return out


def benjamini_hochberg(ps: list[float]) -> list[float]:
    """Benjamini-Hochberg's correction (controls the share of false claims among those made).

    >>> benjamini_hochberg([0.01, 0.04, 0.03])
    [0.03, 0.04, 0.04]
    """
    m = len(ps)
    order = sorted(range(m), key=lambda k: ps[k], reverse=True)
    out, running = [0.0] * m, 1.0
    for i, k in enumerate(order):
        rank = m - i
        running = min(running, ps[k] * m / rank)
        out[k] = round(min(1.0, running), 15)
    return out


def hedges_g(a: list[float], b: list[float]) -> float:
    """The standardised difference in means (Cohen's d with the small-sample correction): 0.2 small, 0.5 medium,
    0.8 large.

    >>> round(hedges_g([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]), 3)  # d = -1 / 1.581, times 1 - 3/(4*8-1)
    -0.571
    """
    sa, sb = Summary(a), Summary(b)
    df = sa.n + sb.n - 2
    if df <= 0:
        return 0.0
    pooled = math.sqrt(((sa.n - 1) * sa.std**2 + (sb.n - 1) * sb.std**2) / df)
    if pooled == 0:
        return 0.0 if sa.mean == sb.mean else math.copysign(math.inf, sa.mean - sb.mean)
    return (sa.mean - sb.mean) / pooled * (1 - 3 / (4 * df - 1))


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """How often a value of a beats one of b, minus the reverse (-1 to 1): an effect size that needs no normality.

    >>> cliffs_delta([3, 4, 5], [1, 2, 3])
    0.888889
    """
    ordered = sorted(b)
    wins = sum(bisect_left(ordered, x) - (len(ordered) - bisect_right(ordered, x)) for x in a)  # below x minus above x
    return round(wins / (len(a) * len(b)), 6)


def seeds_needed(a: list[float], b: list[float], alpha: float = 0.05, power: float = 0.8, paired: bool = False) -> int | None:
    """How many seeds per method a difference of the size seen would need to be found significant (two-sided, with
    the given power), from the spread seen: the smallest n (at least 2) for which the t approximation of the power
    is enough, at n's own degrees of freedom (small samples pay for their few). Paired seeds go by the spread of the
    differences. None when there is no difference or no spread to go by.

    >>> seeds_needed([93.0, 93.4, 92.9], [92.8, 93.1, 92.6])  # n = 16: 2 * ((2.042 + 0.854) * 0.258 / 0.267)^2 = 15.7
    16
    >>> seeds_needed([74.1, 73.8], [70.2, 70.6], paired=True)  # differences 3.9, 3.2: two seeds were not enough
    3
    """
    sa, sb = Summary(a), Summary(b)
    diff = abs(sa.mean - sb.mean)
    spread = Summary([x - y for x, y in zip(a, b, strict=True)]).std if paired else math.sqrt((sa.std**2 + sb.std**2) / 2)
    if diff == 0 or spread == 0 or sa.n < 2 or sb.n < 2:
        return None

    def required(n: int) -> float:
        df = n - 1 if paired else 2 * n - 2
        z = t_ppf(1 - alpha / 2, df) + t_ppf(power, df)
        return (1 if paired else 2) * (z * spread / diff) ** 2

    n = 2
    while required(n) > n:
        n = n + 1 if n < 64 else n * 2  # past 64, an order of magnitude is all one needs to know
        if n > 100_000:
            return None
    return n


def all_pairs(groups: dict[str, list[float]]) -> list[tuple[str, str]]:
    return list(combinations(groups, 2))
