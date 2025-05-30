import pandas as pd
import numpy as np
from scipy.stats import rankdata
import csv
from operator import itemgetter
import os
import webbrowser

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
    print("Calculating weights...")
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

def improvedExperiment():
    df = pd.read_csv("datasets/qws.csv")

    columns = [
        'Response Time', 'Latency', 'Availability', 'Reliability',
        'Best Practices', 'Successability', 'Service Name', 'WSDL Address'
    ]
    df = df[columns].dropna()

    # Separate benefit_criteria and cost_criteria
    benefit_criteria = ['Availability', 'Reliability', 'Best Practices', 'Successability']
    cost_criteria = ['Response Time', 'Latency']

    normalized_df = pd.DataFrame()
    for col in benefit_criteria:
        normalized_df[col] = df[col] / df[col].max()
    for col in cost_criteria:
        normalized_df[col] = df[col].min() / df[col]

    P = normalized_df / normalized_df.sum()
    E = -np.nansum(P * np.log(P + 1e-10), axis=0) / np.log(len(df))
    d = 1 - E
    fuzzy_weights = d / d.sum()

    trust_scores = (normalized_df * fuzzy_weights).sum(axis=1)

    df['Trust Score'] = trust_scores
    top_10 = df.sort_values(by=["Trust Score"], ascending=[False])

    df = df.sort_values(by=["Trust Score"], ascending=[False])

    print("Trusted Web Services by a trust score:")
    print(top_10[['Service Name', 'WSDL Address', 'Trust Score']])

    # Ability to save into CSV (if needed)
    # top_10.to_csv("top10_trusted_services.csv", index=False)

def get_services(filepath):
    result = []
    with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 5: # 5 values
                continue
            serviceName, serviceAddress, waspas, vikor, topsis = row
            result.append((serviceName, serviceAddress, waspas, vikor, topsis));
    return result

def generate_html_report(services, output_path, totalRows):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = """
    <html>
    <head>
        <title>Top 10 Most Trusted Services</title>
        <style>
            body { font-family: Arial, sans-serif; }
            table { border-collapse: collapse; width: 60%; margin: 20px auto; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Top 10 Most Trusted Services</h2>
        <table>
            <tr>
                <th>Service Name</th>
                <th>Service address</th>
                <th>Trust Score</th>
                <th>Waspas score</th>
                <th>Vikor score</th>
                <th>Topsis score</th>
            </tr>
    """ # Basic document, should become a dynamic one in the future
    generated = 0

    for serviceName, serviceAddress, waspas, vikor, topsis in services:
        if serviceName == 'Service Name': continue
        if generated == totalRows: break
        generated += 1
        html += f"""
            <tr>
                <td>{serviceName}</td>
                <td>{serviceAddress}</td>
                <td>{waspas}</td>
                <td>{vikor}
                <td>{topsis}
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html)
    print(f"HTML report saved to {output_path}")
    print(f'Total number of web services saved: {generated}')
    webbrowser.open(f'file://{os.path.abspath(output_path)}') # Fun for a quick test but might be annoying in the longterm


if __name__ == "__main__":
    #main()
    improvedExperiment()
    csv_path = 'datasets/qws_result_trust.csv'
    output_html_path = 'output/trusted_services_report.html'

    services = get_services(csv_path); print(services)
    generate_html_report(services, output_html_path, 10) # to become in a config
print('----- Report generation complete! -----')
