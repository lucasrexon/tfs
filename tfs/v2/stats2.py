"""Correlation and significance tools used by the v2 research."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cvar(x, q: float = 0.05) -> float:
    """Average of the worst q share of outcomes."""
    x = np.sort(np.asarray(x, float))
    k = max(1, int(np.ceil(len(x) * q)))
    return float(x[:k].mean())


def bh(pvals: pd.Series, alpha: float = 0.10) -> pd.Series:
    """Benjamini-Hochberg: which p-values survive a false-discovery rate of alpha."""
    p = pvals.dropna().sort_values()
    m = len(p)
    ok = p <= alpha * (np.arange(1, m + 1) / m)
    cut = p[ok].max() if ok.any() else -1
    return pvals <= cut


def correlate(df: pd.DataFrame, features: list[str], y: str, binary: set[str]) -> pd.DataFrame:
    """Each feature against outcome y (0/1 or a number).

    0/1 feature vs 0/1 outcome: phi coefficient (Pearson on two binaries) + Fisher exact p.
    number vs 0/1 outcome:      point-biserial r (Pearson) and Spearman rho (rank based).
    """
    rows = []
    yb = df[y].nunique() <= 2
    for f in features:
        d = df[[f, y]].dropna()
        if len(d) < 30 or d[f].nunique() < 2:
            continue
        x, t = d[f].astype(float).to_numpy(), d[y].astype(float).to_numpy()
        r, p = stats.pearsonr(x, t)
        row = {"feature": f, "n": len(d), "kind": "yes/no" if f in binary else "number", "r": r, "p": p}
        if f in binary and yb:
            n11 = int(((x == 1) & (t == 1)).sum()); n10 = int(((x == 1) & (t == 0)).sum())
            n01 = int(((x == 0) & (t == 1)).sum()); n00 = int(((x == 0) & (t == 0)).sum())
            row["p"] = stats.fisher_exact([[n11, n10], [n01, n00]])[1]
            row["rate_with"] = n11 / max(1, n11 + n10)
            row["rate_without"] = n01 / max(1, n01 + n00)
            row["n_with"] = n11 + n10
        else:
            rho, ps = stats.spearmanr(x, t)
            row["rho"], row["p"] = rho, ps          # ranks: these features are very skewed
        lo, hi = np.tanh(np.arctanh(np.clip(row.get("rho", r), -0.999, 0.999)) + np.array([-1.96, 1.96]) / np.sqrt(len(d) - 3))
        row["ci_lo"], row["ci_hi"] = lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant"] = bh(out["p"])
        out = out.sort_values("p")
    return out


def lift_table(df: pd.DataFrame, col: str, y: str, min_n: int = 15) -> pd.DataFrame:
    base = df[y].mean()
    rows = []
    for k, g in df.groupby(col):
        n, s = len(g), int(g[y].sum())
        lo, hi = wilson(s, n)
        rows.append({col: k, "n": n, "rate": s / n, "lift": (s / n) / base if base else np.nan,
                     "rate_lo": lo, "rate_hi": hi, "enough": n >= min_n})
    return pd.DataFrame(rows).sort_values("rate", ascending=False)


class Logit:
    """Small ridge logistic regression (IRLS), features standardized, missing -> median."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def _prep(self, X: pd.DataFrame, fit=False):
        if fit:
            self.cols = list(X.columns)
            self.med = X.median()
            Xf = X.fillna(self.med)
            self.mu, self.sd = Xf.mean(), Xf.std(ddof=0).replace(0, 1)
        Z = ((X[self.cols].fillna(self.med) - self.mu) / self.sd).to_numpy(float)
        return np.c_[np.ones(len(Z)), np.clip(Z, -5, 5)]

    def fit(self, X: pd.DataFrame, y):
        A, y = self._prep(X, True), np.asarray(y, float)
        b = np.zeros(A.shape[1])
        pen = np.eye(A.shape[1]) * self.l2
        pen[0, 0] = 0
        for _ in range(50):
            p = 1 / (1 + np.exp(-A @ b))
            W = p * (1 - p) + 1e-9
            step = np.linalg.solve(A.T @ (A * W[:, None]) + pen, A.T @ (y - p) - pen @ b)
            b += step
            if np.abs(step).max() < 1e-8:
                break
        self.b = b
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self._prep(X) @ self.b))

    def weights(self) -> pd.Series:
        return pd.Series(self.b[1:], index=self.cols).sort_values(key=np.abs, ascending=False)


def auc(y, score) -> float:
    y, score = np.asarray(y), np.asarray(score)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
