# -*- coding: utf-8 -*-
"""
Faithful reimplementation of the trust-evaluation pipeline:
  Step 1   Pre-process + correlation analysis.
  Step 2.1 Crisp baseline: min-max normalize (benefit/cost) -> weighted sum -> [0,1].
  Step 2.2-a Fuzzy WASPAS / TOPSIS / VIKOR computed on the SAME normalized values.
  Step 2.2-b Mamdani FIS over the three MCDM scores (27 rules) -> Trust Score Value.
  Step 3   Classify Trusted/Untrusted at 0.5; agreement via Kendall tau-b / Spearman.

"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "dataset", "qws.csv")
OUTDIR = os.path.join(HERE, "real-data", "output")

PE_ATTRS = ["Response Time", "Latency", "Throughput"]
WEIGHTS = {"Response Time": 0.45, "Latency": 0.35, "Throughput": 0.20}
BENEFIT = {"Response Time": False, "Latency": False, "Throughput": True}  # cost / cost / benefit

# triangular MFs (a, b, c); "High" = best performance. Used for labelling.
# Overlapping form: bases extended to the neighbour's peak so
# adjacent terms cross.
MF = {
    "Latency":       {"High": (0, 333.33, 2166.67),  "Medium": (333.33, 2166.67, 4333.33), "Low": (2166.67, 4333.33, 5000)},
    "Response Time": {"High": (37, 862.45, 2513.34), "Medium": (862.45, 2513.34, 4164.23), "Low": (2513.34, 4164.23, 4989)},
    "Throughput":    {"High": (21, 34.33, 41),       "Medium": (7.67, 21, 34.33),          "Low": (1, 7.67, 21)},
}
TERMS = ["Low", "Medium", "High"]
LAMBDA = 0.5
VIKOR_V = 0.5
EPS = 1e-9

# "row" = matching 1-based dataset row
# (confirmed by exact match of the MCDM scores; the "cN" labels are row indices).
EXAMPLES = {
    "c1":    {"row": 1,    "WASPAS": 0.730, "TOPSIS": 0.761, "VIKOR": 0.707, "Trust": 0.566, "Crisp": 0.688},
    "c136":  {"row": 135,  "WASPAS": 0.088, "TOPSIS": 0.122, "VIKOR": 0.112, "Trust": 0.342, "Crisp": 0.0813},
    "c202":  {"row": 201,  "WASPAS": 0.940, "TOPSIS": 0.925, "VIKOR": 0.920, "Trust": 0.723, "Crisp": 0.915},
    "c215":  {"row": 214,  "WASPAS": 0.200, "TOPSIS": 0.314, "VIKOR": 0.169, "Trust": 0.412, "Crisp": 0.242},
    "c375":  {"row": 374,  "WASPAS": 0.017, "TOPSIS": 0.052, "VIKOR": 0.036, "Trust": 0.239, "Crisp": 0.0268},
    "c406":  {"row": 405,  "WASPAS": 0.942, "TOPSIS": 0.930, "VIKOR": 0.924, "Trust": 0.724, "Crisp": 0.918},
    "c1162": {"row": 1161, "WASPAS": 0.999, "TOPSIS": 0.999, "VIKOR": 1.000, "Trust": 0.834, "Crisp": 0.999},
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def trimf(x, abc):
    """Triangular membership of scalar x for triangle (a, b, c). Handles shoulders (a==b, b==c)."""
    a, b, c = abc
    if x <= a:
        return 1.0 if (x == a and a == b) else 0.0
    if x >= c:
        return 1.0 if (x == c and b == c) else 0.0
    if x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def dist_to_triangle(x, abc):
    a, _, c = abc
    if x < a:
        return a - x
    if x > c:
        return x - c
    return 0.0


def label_term(value, attr):
    """Dominant linguistic term under MFs (nearest triangle if in a gap)."""
    mus = {t: trimf(value, MF[attr][t]) for t in TERMS}
    best = max(mus, key=mus.get)
    if mus[best] > 0:
        return best
    return min(TERMS, key=lambda t: dist_to_triangle(value, MF[attr][t]))


def thirds_label(x):
    if x < 1 / 3:
        return "Low"
    if x < 2 / 3:
        return "Moderate"
    return "High"


# --------------------------------------------------------------------------- #
# Data + normalization
# --------------------------------------------------------------------------- #
def load_data():
    df = pd.read_csv(DATASET)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    return df.dropna(subset=PE_ATTRS).reset_index(drop=True)


def normalize(df):
    """A1: min-max per criterion, best = 1."""
    r = pd.DataFrame(index=df.index)
    for attr in PE_ATTRS:
        x = df[attr].astype(float)
        lo, hi = x.min(), x.max()
        rng = hi - lo
        if rng == 0:
            r[attr] = 1.0
        elif BENEFIT[attr]:
            r[attr] = (x - lo) / rng
        else:
            r[attr] = (hi - x) / rng
    return r


# --------------------------------------------------------------------------- #
# correlation analysis
# --------------------------------------------------------------------------- #
def correlation_table(df):
    """Returns Kendall tau-b and Spearman rho matrices over the QoS attributes."""
    cols = [c for c in ["Response Time", "Throughput", "Latency",
                        "Availability", "Successability", "Reliability"] if c in df.columns]
    kendall = pd.DataFrame(index=cols, columns=cols, dtype=float)
    spear = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            mask = df[a].notna() & df[b].notna()
            kendall.loc[a, b] = float(kendalltau(df.loc[mask, a], df.loc[mask, b]).statistic)
            spear.loc[a, b] = float(spearmanr(df.loc[mask, a], df.loc[mask, b]).statistic)
    return kendall, spear


def combine_ks(kendall, spear):
    """Format two correlation matrices as 'kendall / spearman' strings."""
    out = pd.DataFrame(index=kendall.index, columns=kendall.columns, dtype=object)
    for a in kendall.index:
        for b in kendall.columns:
            out.loc[a, b] = f"{kendall.loc[a, b]:.3f}/{spear.loc[a, b]:.3f}"
    return out


# --------------------------------------------------------------------------- #
# crisp baseline
# --------------------------------------------------------------------------- #
def crisp_baseline(r):
    return sum(r[a] * WEIGHTS[a] for a in PE_ATTRS)


# --------------------------------------------------------------------------- #
# MCDM methods (A2 - on normalized values)
# --------------------------------------------------------------------------- #
def fuzzy_waspas(r):
    w = np.array([WEIGHTS[a] for a in PE_ATTRS])
    mat = r[PE_ATTRS].to_numpy()
    wsm = (mat * w).sum(axis=1)
    wpm = np.prod(np.clip(mat, EPS, None) ** w, axis=1)
    return LAMBDA * wsm + (1 - LAMBDA) * wpm


def fuzzy_topsis(r):
    w = np.array([WEIGHTS[a] for a in PE_ATTRS])
    v = r[PE_ATTRS].to_numpy() * w
    ideal = v.max(axis=0)
    anti = v.min(axis=0)
    d_plus = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((v - anti) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    return np.where(denom > 0, d_minus / np.where(denom == 0, 1, denom), 0.0)


def fuzzy_vikor(r):
    w = np.array([WEIGHTS[a] for a in PE_ATTRS])
    mat = r[PE_ATTRS].to_numpy()
    fstar = mat.max(axis=0)   # best (higher = better)
    fminus = mat.min(axis=0)  # worst
    rng = np.where((fstar - fminus) == 0, 1, fstar - fminus)
    contrib = w * (fstar - mat) / rng          # per-criterion regret
    S = contrib.sum(axis=1)
    R = contrib.max(axis=1)
    qs = 0.0 if S.max() == S.min() else (S - S.min()) / (S.max() - S.min())
    qr = 0.0 if R.max() == R.min() else (R - R.min()) / (R.max() - R.min())
    Q = VIKOR_V * qs + (1 - VIKOR_V) * qr
    return 1.0 - Q  # orient: higher = better


# --------------------------------------------------------------------------- #
# Mamdani FIS over the 3 MCDM scores (A6)
# --------------------------------------------------------------------------- #
FIS_IN = {"Low": (0.0, 0.0, 0.5), "Medium": (0.0, 0.5, 1.0), "High": (0.5, 1.0, 1.0)}
FIS_OUT = {"Low": (0.0, 0.0, 0.5), "Moderate": (0.25, 0.5, 0.75), "High": (0.5, 1.0, 1.0)}
_ORD = {"Low": 0, "Medium": 1, "High": 2}


def _build_rules():
    rules = []
    for w in TERMS:
        for t in TERMS:
            for v in TERMS:
                avg = (_ORD[w] + _ORD[t] + _ORD[v]) / 3.0
                out = "Low" if avg <= 2 / 3 else ("Moderate" if avg <= 4 / 3 else "High")
                rules.append(((w, t, v), out))
    return rules


_RULES = _build_rules()
_GRID = np.linspace(0, 1, 201)
_OUT_CURVES = {k: np.array([trimf(z, v) for z in _GRID]) for k, v in FIS_OUT.items()}


def fis_trust(waspas, topsis, vikor):
    mu = {
        "w": {t: trimf(waspas, FIS_IN[t]) for t in TERMS},
        "t": {t: trimf(topsis, FIS_IN[t]) for t in TERMS},
        "v": {t: trimf(vikor, FIS_IN[t]) for t in TERMS},
    }
    strength = {"Low": 0.0, "Moderate": 0.0, "High": 0.0}
    for (w, t, v), out in _RULES:
        s = min(mu["w"][w], mu["t"][t], mu["v"][v])
        if s > strength[out]:
            strength[out] = s
    agg = np.zeros_like(_GRID)
    for term, s in strength.items():
        if s > 0:
            agg = np.maximum(agg, np.minimum(s, _OUT_CURVES[term]))
    denom = agg.sum()
    return float((_GRID * agg).sum() / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    df = load_data()
    print(f"Loaded {len(df)} web services with complete Performance-Efficiency data.\n")

    kendall, spear = correlation_table(df)
    print("=== QoS correlation analysis (Kendall tau-b / Spearman rho) ===")
    print(combine_ks(kendall, spear).to_string(), "\n")

    r = normalize(df)
    for attr in PE_ATTRS:
        df[attr + " (term)"] = df[attr].apply(lambda x, a=attr: label_term(x, a))

    df["Crisp_Score"] = crisp_baseline(r)
    df["Crisp_Label"] = df["Crisp_Score"].map(thirds_label)
    df["WASPAS"] = fuzzy_waspas(r)
    df["TOPSIS"] = fuzzy_topsis(r)
    df["VIKOR"] = fuzzy_vikor(r)
    df["Trust_Score"] = [fis_trust(w, t, v) for w, t, v in zip(df["WASPAS"], df["TOPSIS"], df["VIKOR"])]
    df["Trust_Label"] = df["Trust_Score"].map(thirds_label)
    df["Trusted_Fuzzy"] = np.where(df["Trust_Score"] >= 0.5, "Trusted", "Untrusted")
    df["Trusted_Crisp"] = np.where(df["Crisp_Score"] >= 0.5, "Trusted", "Untrusted")

    tau = kendalltau(df["Trust_Score"], df["Crisp_Score"]).statistic
    rho = spearmanr(df["Trust_Score"], df["Crisp_Score"]).statistic
    n_ft = (df["Trusted_Fuzzy"] == "Trusted").sum()
    n_ct = (df["Trusted_Crisp"] == "Trusted").sum()

    print("=== Agreement between fuzzy-MCDM (FIS) and crisp Trust Scores ===")
    t5 = pd.DataFrame(
        [[f"1.000/1.000", f"{tau:.3f}/{rho:.3f}"], [f"{tau:.3f}/{rho:.3f}", f"1.000/1.000"]],
        index=["Fuzzy-MCDM Trust Score", "Crisp Trust Score"],
        columns=["Fuzzy-MCDM Trust Score", "Crisp Trust Score"])
    print(t5.to_string(), "  (Kendall tau-b / Spearman rho)\n")
    print(f"  Kendall tau-b={tau:.3f}  Spearman rho={rho:.3f}  R^2={rho**2:.3f}   "
          f"[published: 0.943 / 0.993 / 0.986]")
    print(f"  Trusted (fuzzy): {n_ft} / {len(df)}    Trusted (crisp): {n_ct} / {len(df)}")
    print("  Pairwise vs Crisp (Kendall tau-b): " + "  ".join(
        f"{x}={kendalltau(df[x], df['Crisp_Score']).statistic:.3f}"
        for x in ["WASPAS", "TOPSIS", "VIKOR", "Trust_Score"]) + "\n")

    print("=== Result ===")
    print(f"{'case':>6} | {'WASPAS pub/new':>17} | {'TOPSIS pub/new':>17} | "
          f"{'VIKOR pub/new':>17} | {'Trust pub/new':>17} | {'Crisp pub/new':>17} | {'new label':>9}")
    for case, ex in EXAMPLES.items():
        x = df.iloc[ex["row"] - 1]
        print(f"{case:>6} | {ex['WASPAS']:>8.3f}/{x['WASPAS']:<8.3f} | "
              f"{ex['TOPSIS']:>8.3f}/{x['TOPSIS']:<8.3f} | "
              f"{ex['VIKOR']:>8.3f}/{x['VIKOR']:<8.3f} | "
              f"{ex['Trust']:>8.3f}/{x['Trust_Score']:<8.3f} | "
              f"{ex['Crisp']:>8.3f}/{x['Crisp_Score']:<8.3f} | {x['Trust_Label']:>9}")
    print()

    os.makedirs(OUTDIR, exist_ok=True)
    out_cols = (["Service Name"] + PE_ATTRS + [a + " (term)" for a in PE_ATTRS] +
                ["Crisp_Score", "Crisp_Label", "WASPAS", "TOPSIS", "VIKOR",
                 "Trust_Score", "Trust_Label", "Trusted_Fuzzy", "Trusted_Crisp"])
    out_path = os.path.join(OUTDIR, "pipeline_results.csv")
    df[[c for c in out_cols if c in df.columns]].to_csv(out_path, index=False)
    print(f"Per-service results written to: {out_path}")


if __name__ == "__main__":
    main()
