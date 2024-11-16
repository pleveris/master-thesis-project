# -*- coding: utf-8 -*-
"""Fuzzy Logic for QWS Dataset"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables for QWS criteria: availability, reliability, throughput, latency, compliance
availability = ctrl.Antecedent(np.arange(0, 11, 1), 'availability')
reliability = ctrl.Antecedent(np.arange(0, 11, 1), 'reliability')
throughput = ctrl.Antecedent(np.arange(0, 101, 1), 'throughput')
latency = ctrl.Antecedent(np.arange(0, 101, 1), 'latency')
compliance = ctrl.Antecedent(np.arange(0, 11, 1), 'compliance')

quality = ctrl.Consequent(np.arange(0, 101, 1), 'quality')

# Membership functions for predefined criteria
availability.automf(3)  # Low, Medium, High
reliability.automf(3)   # Low, Medium, High

throughput['low'] = fuzz.trimf(throughput.universe, [0, 0, 50])
throughput['medium'] = fuzz.trimf(throughput.universe, [0, 50, 100])
throughput['high'] = fuzz.trimf(throughput.universe, [50, 100, 100])

latency['low'] = fuzz.trimf(latency.universe, [0, 0, 30])
latency['medium'] = fuzz.trimf(latency.universe, [0, 30, 70])
latency['high'] = fuzz.trimf(latency.universe, [30, 70, 100])

compliance.automf(3)  # Low, Medium, High
quality.automf(3)     # Poor, Average, Good

# Rules definition
rules = [
    ctrl.Rule(availability['poor'] & reliability['poor'], quality['poor']),
    ctrl.Rule(availability['average'] & reliability['average'], quality['average']),
    ctrl.Rule(availability['good'] & reliability['good'], quality['good']),
    ctrl.Rule(throughput['low'], quality['poor']),
    ctrl.Rule(throughput['medium'] & latency['low'], quality['average']),
    ctrl.Rule(throughput['high'] & latency['low'], quality['good']),
    ctrl.Rule(latency['high'], quality['poor']),
    ctrl.Rule(compliance['poor'], quality['poor']),
    ctrl.Rule(compliance['average'], quality['average']),
    ctrl.Rule(compliance['good'], quality['good']),
]

# System simulation
qws_comparison = ctrl.ControlSystem(rules)
qws_simulation = ctrl.ControlSystemSimulation(qws_comparison)

# Sample inputs
inputs = {
    'availability': 7,
    'reliability': 8,
    'throughput': 60,
    'latency': 20,
    'compliance': 9
}

for key, value in inputs.items():
    qws_simulation.input[key] = value

qws_simulation.compute()

print("\nInput Criteria:")
for key, value in inputs.items():
    print(f"{key.capitalize()}: {value}")
print(f"\nCalculated Quality of Web Service: {qws_simulation.output['quality']:.2f}")

# Display the membership functions
availability.view(sim=qws_simulation)
reliability.view(sim=qws_simulation)
throughput.view(sim=qws_simulation)
latency.view(sim=qws_simulation)
compliance.view(sim=qws_simulation)
quality.view(sim=qws_simulation)
