# -*- coding: utf-8 -*-
"""Fuzzy for Web Services - adapted from the sample Colab project"""

#pip install -U -q scikit-fuzzy

"""# Fuzzy experiments for web service selection"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
web_service_availability = ctrl.Antecedent(np.arange(0, 11, 1), 'web_service_availability')
web_service_reliability = ctrl.Antecedent(np.arange(0, 11, 1), 'web_service_reliability')
web_service_response_time = ctrl.Consequent(np.arange(0, 101, 1), 'web_service_response_time')

# Define membership functions
web_service_availability['low'] = fuzz.trimf(web_service_availability.universe, [0, 0, 5])
web_service_availability['medium'] = fuzz.trimf(web_service_availability.universe, [0, 5, 10])
web_service_availability['high'] = fuzz.trimf(web_service_availability.universe, [5, 10, 10])

web_service_reliability['low'] = fuzz.trimf(web_service_reliability.universe, [0, 0, 5])
web_service_reliability['medium'] = fuzz.trimf(web_service_reliability.universe, [0, 5, 10])
web_service_reliability['high'] = fuzz.trimf(web_service_reliability.universe, [5, 10, 10])

web_service_response_time['low'] = fuzz.trimf(web_service_response_time.universe, [0, 0, 50])
web_service_response_time['medium'] = fuzz.trimf(web_service_response_time.universe, [0, 50, 100])
web_service_response_time['high'] = fuzz.trimf(web_service_response_time.universe, [50, 100, 100])

# Define rules
rule1 = ctrl.Rule(web_service_availability['low'] & web_service_reliability['low'], web_service_response_time['low'])
rule2 = ctrl.Rule(web_service_availability['medium'] & web_service_reliability['low'], web_service_response_time['medium'])
rule3 = ctrl.Rule(web_service_availability['high'] & web_service_reliability['low'], web_service_response_time['high'])
rule4 = ctrl.Rule(web_service_availability['low'] & web_service_reliability['medium'], web_service_response_time['high'])
rule5 = ctrl.Rule(web_service_availability['medium'] & web_service_reliability['medium'], web_service_response_time['medium'])
rule6 = ctrl.Rule(web_service_availability['high'] & web_service_reliability['medium'], web_service_response_time['high'])
rule7 = ctrl.Rule(web_service_availability['low'] & web_service_reliability['high'], web_service_response_time['low'])
rule8 = ctrl.Rule(web_service_availability['medium'] & web_service_reliability['high'], web_service_response_time['medium'])
rule9 = ctrl.Rule(web_service_availability['high'] & web_service_reliability['high'], web_service_response_time['high'])

# Create web service comparison system
web_service_comparison = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
web_service_simulation = ctrl.ControlSystemSimulation(web_service_comparison)

# Main loop for user input and output
while True:
    print("\nWeb service comparison system")
    print("1. Compare web services between different criteria")
    print("2. Exit")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # User input for availability percentage
        ws_availability_percentage_input = float(input("Enter availability percentage (0-10, increases by 10%): "))
        while ws_availability_percentage_input < 0 or ws_availability_percentage_input > 10:
            print("Invalid input! Availability percentage value should be between 0 and 10.")
            ws_availability_percentage_input = float(input("Enter availability percentage (0-10, increases by 10%): "))

        # User input for reliability percentage
        ws_reliability_percentage_input = float(input("Enter web service reliability percentage (0-10), increases by 10%: "))
        while ws_reliability_percentage_input < 0 or ws_reliability_percentage_input > 10:
            print("Invalid input! Reliability percentage value should be between 0 and 10.")
            ws_reliability_percentage_input = float(input("Enter web service reliability percentage (0-10), increases by 10%: "))

        # Set input values
        web_service_simulation.input['web_service_availability'] = ws_availability_percentage_input
        web_service_simulation.input['web_service_reliability'] = ws_reliability_percentage_input

        # Compute the result
        web_service_simulation.compute()

        # Output result
        print("\nSimulation Results:")
        print("Web service response time :", web_service_simulation.output['web_service_response_time'])
        print("Web service availability:", ws_availability_percentage_input)
        print("Web service reliability:", ws_reliability_percentage_input)

        # Display fuzzy logic matrices in tabular form
        print("\nFuzzy Logic Matrix for web service availability:")
        print("Availability\tLow\tMedium\tHigh")
        for val in range(11):
            print(f"{val}\t\t{fuzz.interp_membership(web_service_availability.universe, web_service_availability['low'].mf, val):.2f}\t{fuzz.interp_membership(web_service_availability.universe, web_service_availability['medium'].mf, val):.2f}\t{fuzz.interp_membership(web_service_availability.universe, web_service_availability['high'].mf, val):.2f}")

        print("\nFuzzy Logic Matrix for web service reliability:")
        print("Reliability\tLow\tMedium\tHigh")
        for val in range(11):
            print(f"{val}\t\t{fuzz.interp_membership(web_service_reliability.universe, web_service_reliability['low'].mf, val):.2f}\t{fuzz.interp_membership(web_service_reliability.universe, web_service_reliability['medium'].mf, val):.2f}\t{fuzz.interp_membership(web_service_reliability.universe, web_service_reliability['high'].mf, val):.2f}")

        print("\nFuzzy Logic Matrix for web service response time:")
        print("Response time\tLow\tMedium\High")
        for val in range(101):
            print(f"{val}\t\t{fuzz.interp_membership(web_service_response_time.universe, web_service_response_time['low'].mf, val):.2f}\t{fuzz.interp_membership(web_service_response_time.universe, web_service_response_time['medium'].mf, val):.2f}\t{fuzz.interp_membership(web_service_response_time.universe, web_service_response_time['high'].mf, val):.2f}")

        # Display output graphs
        web_service_availability.view(sim=web_service_simulation)
        web_service_reliability.view(sim=web_service_simulation)
        web_service_response_time.view(sim=web_service_simulation)

    elif choice == '2':
        print("Exiting the program.")
        break

    else:
        print("Invalid choice! Please enter 1 or 2.")
