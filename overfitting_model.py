import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self):
        full_df = pd.read_csv("/storage/affectnet+expw.csv")
        large_df, small_df = train_test_split(full_df, test_size=0.1)
        self.train_df, self.test_df = train_test_split(small_df, test_size=0.25)
    def get_data(self):
        return (self.train_df, self.test_df)

if __name__ == "__main__":
    print(DataManager().get_data())
