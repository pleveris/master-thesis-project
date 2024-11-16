# -*- coding: utf-8 -*-
"""QWS Trustworthiness Evaluation using Fuzzy Logic"""
# Initial version 2024-11-16
# Author: Paulius Leveris <paulius.leveris@gmail.com>

from src import GlobalVars
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load data
csv_file = GlobalVars.dataset_path
columns = [
    'Response Time', 'Availability', 'Throughput', 'Successability',
    'Reliability', 'Compliance', 'Best Practices', 'Latency',
    'Documentation', 'Service Name', 'WSDL Address'
]
qws_data = pd.read_csv(csv_file, names=columns)

# Define fuzzy variables
response_time = ctrl.Antecedent(np.arange(0, 4000, 1), 'response_time')
availability = ctrl.Antecedent(np.arange(0, 101, 1), 'availability')
throughput = ctrl.Antecedent(np.arange(0, 101, 1), 'throughput')
reliability = ctrl.Antecedent(np.arange(0, 101, 1), 'reliability')
trustworthiness = ctrl.Consequent(np.arange(0, 101, 1), 'trustworthiness')

# Define membership functions
response_time['fast'] = fuzz.trimf(response_time.universe, [0, 0, 500])
response_time['medium'] = fuzz.trimf(response_time.universe, [500, 1500, 3000])
response_time['slow'] = fuzz.trimf(response_time.universe, [1500, 4000, 4000])

availability.automf(3)  # Low, Medium, High
throughput.automf(3)    # Low, Medium, High
reliability.automf(3)   # Low, Medium, High
trustworthiness.automf(3)  # Low, Medium, High

# Define fuzzy rules
rules = [
    ctrl.Rule(response_time['fast'] & availability['good'] & reliability['good'], trustworthiness['good']),
    ctrl.Rule(response_time['medium'] & availability['average'] & reliability['average'], trustworthiness['average']),
    ctrl.Rule(response_time['slow'] | availability['poor'], trustworthiness['poor']),
    ctrl.Rule(throughput['high'] & availability['good'], trustworthiness['good']),
    ctrl.Rule(throughput['low'], trustworthiness['poor']),
]

# Create and simulate the fuzzy control system
trust_control_system = ctrl.ControlSystem(rules)
trust_simulation = ctrl.ControlSystemSimulation(trust_control_system)

# Evaluate trustworthiness for each web service
def evaluate_trustworthiness(row):
    try:
        trust_simulation.input['response_time'] = row['Response Time']
        trust_simulation.input['availability'] = row['Availability']
        trust_simulation.input['throughput'] = row['Throughput']
        trust_simulation.input['reliability'] = row['Reliability']
        trust_simulation.compute()
        return trust_simulation.output['trustworthiness']
    except Exception as e:
        print(f"Error processing row {row['Service Name']}: {e}")
        return 0

# Apply the evaluation to the dataset
qws_data['Trustworthiness'] = qws_data.apply(evaluate_trustworthiness, axis=1)

qws_data_sorted = qws_data.sort_values(by='Trustworthiness', ascending=False)

# Display the top 10 most trustworthy services
print("\nTop 10 Most Trustworthy Web Services:")
print(qws_data_sorted[['Service Name', 'Trustworthiness']].head(10))

# Save the sorted data to a CSV file
qws_data_sorted.to_csv('qws_trustworthiness_evaluation.csv', index=False)
