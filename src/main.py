"""
Master Thesis Project - QWS Dataset Analysis Based On Fuzzy Logic Principles
Author: Paulius Lėveris
"""

from Classification import Classification
from FuzzyTopsis import FuzzyTopsis

def main():
    classifier = Classification()
    
    classifier.loadData()
    classifier.process()
    classifier.trainModel()
    classifier.evaluateModel()

    # As a test, perform FUZZY TOPSIS analysis on the first 15 services based on Availability NF-criteria
    fuzzy_topsis = FuzzyTopsis(classifier.initialData)
    fuzzy_topsis.evaluate()

if __name__ == "__main__":
    main()
