# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'datasets', 'qws.csv')

df = pd.read_csv(dataset_path)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

columns = [
    'Response Time', 'Availability', 'Throughput', 'Successability',
    'Reliability', 'Compliance', 'Best Practices', 'Latency',
    'Documentation', 'Service Name', 'WSDL Address'
]
df.columns = columns

norm_columns = [
    "Availability", "Throughput", "Successability", "Reliability",
    "Compliance", "Best Practices", "Latency", "Documentation", "Response Time"
]

max_values = df[norm_columns].max()
min_values = df[norm_columns].min()

fuzzy_weights = {
    "Availability": 0.101937,
    "Latency": 0.560578,
    "Compliance": 0.0,
    "Throughput": 0.1092,
    "Successability": 0.0,
    "Best Practices": 0.012625,
}

beneficial = {
    "Availability": True,
    "Throughput": True,
    "Successability": True,
    "Reliability": True,
    "Compliance": True,
    "Best Practices": True,
    "Latency": False,        # LOWER IS BETTER
    "Documentation": True,
    "Response Time": False   # LOWER IS BETTER
}

def normalize(value, attr):
    if pd.isna(value):
        return np.nan
    if beneficial[attr]:            # Maximize
        return value / max_values[attr]
    else:
        return min_values[attr] / value if value != 0 else 0

for attr in fuzzy_weights.keys():
    df[attr + "_Norm"] = df[attr].apply(lambda x: normalize(x, attr))

for attr in fuzzy_weights.keys():
    df[attr + "_Weighted"] = df[attr + "_Norm"] * fuzzy_weights[attr]

weighted_cols = [attr + "_Weighted" for attr in fuzzy_weights.keys()]
df["WSM"] = df[weighted_cols].sum(axis=1)
df["WPM"] = df[weighted_cols].replace(0, 1e-6).prod(axis=1)
df["WASPAS"] = (df["WSM"] + df["WPM"]) / 2

output_cols = ["Service Name", "WSDL Address"]
for attr in fuzzy_weights.keys():
    output_cols += [attr + "_Norm", attr + "_Weighted"]
output_cols += ["WSM", "WPM", "WASPAS"]

report_df = df[output_cols].dropna()
report_df.to_csv("qws_trust_report.csv", index=False)

print("Report generated")
