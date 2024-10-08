"""
Master Thesis Project - QWS Dataset Analysis Based On Fuzzy Logic Principles
Author: Paulius Lėveris
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from DataReader import DataReader

class Classification:
    def __init__(self):
        self.data = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def loadData(self):
        dataReader = DataReader()
        self.data = dataReader.read()
        
        # Display the first few rows to check if data is present
        print(self.data.head())

    def process(self):
        # Some columns are not needed as they do not provide any value (since these are text-only)
        self.data = self.data.iloc[:, :-1]  # Remove trailing empty column if present
        self.data.drop(['Service Name', 'WSDL Address'], axis=1, inplace=True)
        
        # Conversion to numeric values
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        
        # Handle missing values (if any)
        self.data.fillna(self.data.mean(), inplace=True)

        # target label for classification, for testing purposes now: Availability
        threshold = 85  # TODO: configure the value via config or GUI per-user decision
        self.data['Class'] = self.data['Availability'].apply(lambda x: 'High' if x >= threshold else 'Low')

        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        print("Data preprocessing completed.")

    def trainModel(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def evaluateModel(self):
        y_pred = self.model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Vizualizavimas [TODO???]
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Test') # TODO
        plt.xlabel('Test') # TODO
        plt.show()

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
