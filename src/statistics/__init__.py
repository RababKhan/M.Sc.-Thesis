"""Seed-level paired statistics (master protocol section 9); no SciPy dependency.

sign_flip_p    exact two-sided sign-flip permutation p for the mean paired
               difference over all 2^n sign patterns (meet-in-the-middle)
wilcoxon_p     exact two-sided Wilcoxon signed-rank p (zeros dropped, average
               ranks) by dynamic programming over doubled ranks
t_quantile     exact Student-t quantile (regularised incomplete beta + bisection);
               the historical scripts used 3-decimal tables of the same values
describe       mean/median/SD, 95% t and bootstrap intervals, d_z, both p-values
holm           Holm step-down adjusted p-values
mcnemar        exact two-sided McNemar test on paired per-sample correctness
"""
import math

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ exact tests
def _half_sums(d):
    m = len(d)
    signs = 1 - 2 * ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1)
    return signs @ d


def sign_flip_p(d):
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.sum())
    if obs == 0:
        return 1.0
    thr = obs - 1e-9 * max(1.0, obs)
    a, b = _half_sums(d[: n // 2]), np.sort(_half_sums(d[n // 2:]))
    upper = len(b) - np.searchsorted(b, thr - a, side="left")
    lower = np.searchsorted(b, -thr - a, side="right")
    return float((upper.sum() + lower.sum()) / 2.0 ** n)


def sign_flip_p_bruteforce(d):
    d = np.asarray(d, float)
    obs = abs(d.sum())
    if obs == 0:
        return 1.0
    sums = np.abs(_half_sums(d))
    return float((sums >= obs - 1e-9 * max(1.0, obs)).mean())


def wilcoxon_p(d):
    d = np.asarray([x for x in d if abs(x) > 1e-12], float)
    if len(d) == 0:
        return 1.0
    r2 = np.round(2 * pd.Series(np.abs(d)).rank().to_numpy()).astype(int)
    counts = np.zeros(r2.sum() + 1)
    counts[0] = 1
    for r in r2:
        counts[r:] = counts[r:] + counts[: len(counts) - r].copy() if r else counts
    counts /= counts.sum()
    centre = r2.sum() / 2
    w = r2[d > 0].sum()
    values = np.arange(len(counts))
    return float(counts[np.abs(values - centre) >= abs(w - centre) - 1e-9].sum())


# ------------------------------------------------------------------ Student t
def _betacf(a, b, x, eps=1e-15, itmax=500):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, itmax + 1):
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
            return h
    raise RuntimeError("incomplete beta did not converge")


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ------------------------------------------------------------------ summaries
def describe(d, rng=None, boots=10000):
    d = np.asarray(d, float)
    n = len(d)
    rng = rng if rng is not None else np.random.default_rng(0)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    half = t_quantile(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
    means = d[rng.integers(0, n, (boots, n))].mean(axis=1)
    return {"n_seeds": n, "mean_diff": d.mean(), "median_diff": float(np.median(d)), "sd_diff": sd,
            "t_ci95_low": d.mean() - half, "t_ci95_high": d.mean() + half,
            "boot_ci95_low": float(np.percentile(means, 2.5)), "boot_ci95_high": float(np.percentile(means, 97.5)),
            "cohens_dz": d.mean() / sd if n > 1 and sd > 0 else float("nan"),
            "sign_flip_p": sign_flip_p(d), "wilcoxon_p": wilcoxon_p(d),
            "n_positive": int((d > 1e-12).sum()), "n_negative": int((d < -1e-12).sum()),
            "n_ties": int((np.abs(d) <= 1e-12).sum())}


def holm(p):
    p = np.asarray(p, float)
    out, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out


# ------------------------------------------------------------------ McNemar
def _log_binomial_tail(n, k):
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def mcnemar(correct_a, correct_b):
    """(b, c, chi2 with continuity correction, exact two-sided p); b = only A correct."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / n
    log_p = math.log(2.0) + _log_binomial_tail(n, min(b, c)) - n * math.log(2.0)
    return b, c, chi2, (min(1.0, math.exp(log_p)) if log_p > -745.0 else 0.0)
