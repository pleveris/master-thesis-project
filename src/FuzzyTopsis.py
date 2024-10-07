"""
Master Thesis Project - QWS Dataset Analysis Based On Fuzzy Logic Principles
Author: Paulius Lėveris
"""

import pandas as pd
import numpy as np

class FuzzyTopsis:
    def __init__(self, data):
        # For now take first 15 rows from the dataset (as a test)
        self.data = data.head(15)
        self.availability_scores = self.data['Availability'].values

    def process(self):
        """Converts availability scores to fuzzy values (Low, Medium, High)."""
        fuzzy_scores = []
        for score in self.availability_scores:
            if score < 70:
                fuzzy_scores.append((1.0, 0.0, 0.0))  # Low
            elif 70 <= score <= 85:
                fuzzy_scores.append((0.0, 1.0, 0.0))  # Medium
            else:
                fuzzy_scores.append((0.0, 0.0, 1.0))  # High
        self.fuzzy_scores = np.array(fuzzy_scores)
        print("Fuzzification of availability scores completed.")

    def applyFuzzyTopsis(self):
        # Define ideal solutions for Low, Medium, High
        ideal_solution = np.array([0.0, 0.0, 1.0])  # Ideal -> High availability
        anti_ideal_solution = np.array([1.0, 0.0, 0.0])  # Anti-Ideal -> Low availability
        
        distances_to_ideal = np.linalg.norm(self.fuzzy_scores - ideal_solution, axis=1)
        distances_to_anti_ideal = np.linalg.norm(self.fuzzy_scores - anti_ideal_solution, axis=1)
        
        closeness_coefficients = distances_to_anti_ideal / (distances_to_ideal + distances_to_anti_ideal)
        
        # Services ranking
        self.data['FuzzyTOPSIS_Score'] = closeness_coefficients
        self.data = self.data.sort_values(by='FuzzyTOPSIS_Score', ascending=False)
        
        print("FUZZY TOPSIS evaluation and ranking completed.")
        print(self.data[['Service Name', 'Availability', 'FuzzyTOPSIS_Score']])

    def evaluate(self):
        self.process()
        self.applyFuzzyTopsis()
