import pandas as pd
import numpy as np
from scipy.stats import rankdata


def load_dataset(dataset_choice, file_path):
    """
    Load dataset based on user choice (QWS or Custom).
    """
    if dataset_choice.lower() == "qws":
        print("Loading QWS dataset...")
    else:
        print("Loading custom dataset...")
    return pd.read_csv(file_path)

def validate_data(df):
    """
    Validate the dataset by removing duplicates and null values.
    """
    print("Validating dataset...")
    df = df.dropna().drop_duplicates()
    print("Dataset validation complete.")
    return df


def calculate_weights(decision_matrix, criteria_types):
    """
    Calculate weights using the Entropy Weighting Method.
    """
    print("Calculating weights using Entropy Weighting Method...")
    normalized_matrix = decision_matrix.copy()
    for i, col in enumerate(decision_matrix.columns):
        if criteria_types[i] == "max":
            normalized_matrix[col] = decision_matrix[col] / decision_matrix[col].max()
        elif criteria_types[i] == "min":
            normalized_matrix[col] = decision_matrix[col].min() / decision_matrix[col]
    epsilon = 1e-10
    p = normalized_matrix / normalized_matrix.sum(axis=0)
    entropy = -np.nansum(p * np.log(p + epsilon), axis=0) / np.log(len(decision_matrix))
    d = 1 - entropy
    weights = d / d.sum()
    print("Weights calculated successfully.")
    return weights


def infer_criteria_types(decision_matrix):
    """
    Infer whether each criterion is a maximization or minimization type.
    """
    print("Inferring criteria types dynamically...")
    criteria_types = []
    for col in decision_matrix.columns:
        if col.lower() in ["response time", "latency"]:
            criteria_types.append("min")
        else:
            criteria_types.append("max")
    print("Criteria types inferred successfully.")
    return criteria_types


def fuzzy_waspas(df, weights, criteria_types):
    """
    Compute WASPAS scores and rankings based on the input decision matrix.
    """
    print("Applying Fuzzy WASPAS method...")
    normalized_df = df.copy()
    for i, col in enumerate(df.columns):
        if criteria_types[i] == "max":
            normalized_df[col] = df[col] / df[col].max()
        elif criteria_types[i] == "min":
            normalized_df[col] = df[col].min() / df[col]
    weighted_df = normalized_df * weights
    wsm_scores = weighted_df.sum(axis=1)
    wpm_scores = np.prod(np.power(normalized_df, weights), axis=1)
    lambda_param = 0.5
    waspas_scores = lambda_param * wsm_scores + (1 - lambda_param) * wpm_scores
    rankings = rankdata(-waspas_scores, method="dense")
    print("Fuzzy WASPAS method applied successfully.")
    return pd.DataFrame({"WASPAS Score": waspas_scores, "WASPAS Rank": rankings})


def fuzzy_vikor(df, weights, criteria_types):
    """
    Compute VIKOR scores and rankings based on the input decision matrix.
    """
    print("Applying Fuzzy VIKOR method...")
    df = df.copy()
    ideal = []
    anti_ideal = []
    for i, col in enumerate(df.columns):
        if criteria_types[i] == "max":
            ideal.append(df[col].max())
            anti_ideal.append(df[col].min())
        elif criteria_types[i] == "min":
            ideal.append(df[col].min())
            anti_ideal.append(df[col].max())

    si = []
    ri = []
    for i in range(len(df)):
        s = 0
        r = 0
        for j, col in enumerate(df.columns):
            denominator = abs(anti_ideal[j] - ideal[j])
            if denominator == 0:
                normalized_diff = 0
            else:
                normalized_diff = abs(df.iloc[i][col] - ideal[j]) / denominator
            weighted_diff = weights[j] * normalized_diff
            s += weighted_diff
            r = max(r, weighted_diff)
        si.append(s)
        ri.append(r)

    v = 0.5  # Compromise parameter
    min_s, max_s = min(si), max(si)
    min_r, max_r = min(ri), max(ri)
    q = [
        v * (s - min_s) / (max_s - min_s) + (1 - v) * (r - min_r) / (max_r - min_r)
        if (max_s - min_s) != 0 and (max_r - min_r) != 0 else 0
        for s, r in zip(si, ri)
    ]
    rankings = rankdata(q, method="dense")
    print("Fuzzy VIKOR method applied successfully.")
    return pd.DataFrame({"VIKOR Score": q, "VIKOR Rank": rankings})


def generate_report(df, waspas_results, vikor_results):
    """
    Generate a report combining WASPAS and VIKOR results with the original dataset.
    """
    print("Generating report...")
    combined_df = pd.concat([df.reset_index(drop=True), waspas_results, vikor_results], axis=1)
    combined_df.to_csv("evaluation_report.csv", index=False)
    print("Report saved as evaluation_report.csv")
    return combined_df


def main():
    """
    Main function to run the QoS Evaluation Tool.
    """
    print("Welcome to the QoS Evaluation Tool!")
    dataset_choice = input("Choose dataset type (QWS or Custom): ").strip()
    file_path = input("Enter the path to the dataset file: ").strip()
    df = load_dataset(dataset_choice, file_path)
    df = validate_data(df)
    decision_matrix = df.select_dtypes(include=[np.number])
    criteria_types = infer_criteria_types(decision_matrix)
    weights = calculate_weights(decision_matrix, criteria_types)
    waspas_results = fuzzy_waspas(decision_matrix, weights, criteria_types)
    vikor_results = fuzzy_vikor(decision_matrix, weights, criteria_types)
    generate_report(df, waspas_results, vikor_results)


if __name__ == "__main__":
    main()
