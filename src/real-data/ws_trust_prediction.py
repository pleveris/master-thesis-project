# -*- coding: utf-8 -*-
"""QWS Trustworthiness Evaluation using Fuzzy Logic"""
# Initial version 2024-11-16
# Author: Paulius Leveris <paulius.leveris@gmail.com>

from src import GlobalVars
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Load data
csv_file = GlobalVars.dataset_path
columns = [
    'Response Time', 'Availability', 'Throughput', 'Successability',
    'Reliability', 'Compliance', 'Best Practices', 'Latency',
    'Documentation', 'Service Name', 'WSDL Address'
]
qws_data = pd.read_csv(csv_file)
qws_data = qws_data.loc[:, ~qws_data.columns.str.contains('^Unnamed')]

### for debugging purposes (to be removed later)
#for index, row in qws_data.head(5).iterrows():
    #print(f"Row {index}:")
    #for col in qws_data.columns:
        #print(f"  {col}: {row[col]}")

# Define fuzzy variables
response_time = ctrl.Antecedent(np.arange(0, 5001, 1), 'response_time')
availability = ctrl.Antecedent(np.arange(0, 101, 1), 'availability')
throughput = ctrl.Antecedent(np.arange(0, 101, 1), 'throughput')
throughput['low'] = fuzz.trimf(throughput.universe, [0, 0, 30])
throughput['average'] = fuzz.trimf(throughput.universe, [20, 50, 80])
throughput['high'] = fuzz.trimf(throughput.universe, [70, 100, 100])
reliability = ctrl.Antecedent(np.arange(0, 101, 1), 'reliability')
trustworthiness = ctrl.Consequent(np.arange(0, 101, 1), 'trustworthiness')
trustworthiness['poor'] = fuzz.trimf(trustworthiness.universe, [0, 0, 50])
trustworthiness['average'] = fuzz.trimf(trustworthiness.universe, [0, 50, 100])
trustworthiness['good'] = fuzz.trimf(trustworthiness.universe, [50, 100, 100])

# Define membership functions
response_time['fast'] = fuzz.trimf(response_time.universe, [0, 0, 1000])
response_time['medium'] = fuzz.trimf(response_time.universe, [500, 2500, 4000])
response_time['slow'] = fuzz.trimf(response_time.universe, [3000, 5000, 5000])

availability.automf(3)  # Low, Medium, High
#throughput.automf(3)    # Low, Medium, High
reliability.automf(3)   # Low, Medium, High
#trustworthiness.automf(3)  # Low, Medium, High

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
        if not (0 <= row['Response Time'] < 5000):
            raise ValueError(f"Response Time {row['Response Time']} out of range")
        if not (0 <= row['Availability'] <= 100):
            raise ValueError(f"Availability {row['Availability']} out of range")
        if not (0 <= row['Throughput'] <= 100):
            raise ValueError(f"Throughput {row['Throughput']} out of range")
        if not (0 <= row['Reliability'] <= 100):
            raise ValueError(f"Reliability {row['Reliability']} out of range")

        trust_simulation.input['response_time'] = row['Response Time']
        trust_simulation.input['availability'] = row['Availability']
        trust_simulation.input['throughput'] = row['Throughput']
        trust_simulation.input['reliability'] = row['Reliability']
        trust_simulation.compute()
        return trust_simulation.output['trustworthiness']
    except Exception as e:
        print(f"Error processing row {row['Service Name']}: {e}")
        return 0

# Script for QoS Testing
def check_qos(services):
    results = []
    for service in services:
        url = service.get("url")
        service_name = service.get("name")
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # in milliseconds
            availability = 1 if response.status_code == 200 else 0
            throughput = len(response.content) / response_time if response_time > 0 else 0
            
            results.append({
                "Service Name": service_name,
                "URL": url,
                "Response Time (ms)": round(response_time, 2),
                "Availability": availability,
                "Throughput (KB/s)": round(throughput, 2)
            })
        
        except requests.exceptions.RequestException as e:
            results.append({
                "Service Name": service_name,
                "URL": url,
                "Response Time (ms)": None,
                "Availability": 0,
                "Throughput (KB/s)": None,
                "Error": str(e)
            })
    return results

def save_results_to_csv(results, filename="qos_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

# Apply the evaluation to the dataset
qws_data['Trustworthiness'] = qws_data.apply(evaluate_trustworthiness, axis=1)

qws_data_sorted = qws_data.sort_values(by='Trustworthiness', ascending=False)

# Display the top 10 most trustworthy services
print("\nTop 10 Most Trustworthy Web Services:")
print(qws_data_sorted[['Service Name', 'Trustworthiness']].head(10))

# Save the sorted data to a CSV file
qws_data_sorted.to_csv('qws_trustworthiness_evaluation.csv', index=False)

# show the visual representation
plt.figure(figsize=(12, 8))
plt.barh(qws_data_sorted['Service Name'].head(10), qws_data_sorted['Trustworthiness'].head(10), color='skyblue')
plt.xlabel('Trustworthiness Score')
plt.ylabel('Service Name')
plt.title('Top 10 Most Trustworthy Web Services')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('trustworthiness_chart.png')
plt.show()
