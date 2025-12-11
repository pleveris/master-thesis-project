# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'datasets', 'qws.csv')

df = pd.read_csv(dataset_path)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.columns = [
    "ResponseTime", "Availability", "Throughput", "Successability", "Reliability",
    "Compliance", "BestPractices", "Latency", "Documentation",
    "ServiceName", "WSDL"
]

required_attrs = [
    "Availability", "Throughput", "Successability", "Reliability",
    "Compliance", "BestPractices", "Latency", "ResponseTime"
]
df = df.dropna(subset=required_attrs)

fuzzy_weights = {
    "Availability": 0.7,
    "Latency": 0.4,
    "Compliance": 0.8,
    "Throughput": 0.6,
    "Successability": 0.3,
    "BestPractices": 0.2,
}

beneficial = {
    "Availability": True,
    "Throughput": True,
    "Successability": True,
    "Reliability": True,
    "Compliance": True,
    "BestPractices": True,
    "Latency": False,
    "ResponseTime": False
}

for attr in fuzzy_weights:
    if beneficial[attr]:
        df[attr + "_Norm"] = df[attr] / df[attr].max()
    else:
        df[attr + "_Norm"] = df[attr].min() / df[attr]

for attr in fuzzy_weights:
    df[attr + "_Weighted"] = df[attr + "_Norm"] * fuzzy_weights[attr]

weighted_cols = [attr + "_Weighted" for attr in fuzzy_weights]
df["WSM"] = df[weighted_cols].sum(axis=1)
df["WPM"] = df[weighted_cols].replace(0, 1e-6).prod(axis=1)
df["F_WASPAS"] = (df["WSM"] + df["WPM"]) / 2

# ---------- Fuzzy TOPSIS ----------
ideal = df[weighted_cols].max()
anti_ideal = df[weighted_cols].min()

df["D_plus"] = np.sqrt(((df[weighted_cols] - ideal)**2).sum(axis=1))
df["D_minus"] = np.sqrt(((df[weighted_cols] - anti_ideal)**2).sum(axis=1))

df["F_TOPSIS"] = df["D_minus"] / (df["D_plus"] + df["D_minus"])

# ---------- Fuzzy VIKOR ----------
best = df[weighted_cols].max()
worst = df[weighted_cols].min()

S = []
R = []

for i, row in df[weighted_cols].iterrows():
    s_i = 0
    r_i = -np.inf
    for attr in fuzzy_weights:
        fi = row[attr + "_Weighted"]
        fi_star = best[attr + "_Weighted"]
        fi_minus = worst[attr + "_Weighted"]
        weight = fuzzy_weights[attr]
        score = weight * abs(fi_star - fi) / (fi_star - fi_minus + 1e-6)
        s_i += score
        r_i = max(r_i, score)
    S.append(s_i)
    R.append(r_i)

df["S"] = S
df["R"] = R

# ---------- Fuzzy WASPAS ----------

v = 0.5
S_star, S_minus = min(S), max(S)
R_star, R_minus = min(R), max(R)

df["F_VIKOR"] = v * (df["S"] - S_star) / (S_minus - S_star + 1e-6) + \
                (1 - v) * (df["R"] - R_star) / (R_minus - R_star + 1e-6)

columns = [
    "ServiceName", "WSDL"
]
for attr in fuzzy_weights:
    columns.append(attr + "_Norm")
    columns.append(attr + "_Weighted")

columns += ["WSM", "WPM", "F_WASPAS", "F_TOPSIS", "F_VIKOR"]
df[columns].to_csv("qws-webservices-report.csv", index=False)

print("Report generated")
