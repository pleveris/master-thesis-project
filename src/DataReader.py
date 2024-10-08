"""
Master Thesis Project - QWS Dataset Analysis Based On Fuzzy Logic Principles
Author: Paulius Lėveris
"""

import pandas as pd
import GlobalVars

class DataReader:
    def __init__(self):
        self.dataset_path = GlobalVars.dataset_path

    def read(self) -> str:
        """Returns the whole CSV file data"""
        return pd.read_csv(self.dataset_path)
