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

class Generators:
    def __init__(self, train_df, test_df):
        generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
        self.train_generator = generator.flow_from_dataframe(
            directory = "/storage/facial_expression_images/images",
            dataframe = train_df,
            x_col="filenames",
            y_col="expression",
            target_size=(32, 32),
            class_mode='categorical',
            batch_size=256
        )
        self.test_generator = generator.flow_from_dataframe(
            directory = "/storage/facial_expression_images/images",
            dataframe = test_df,
            x_col="filenames",
            y_col="expression",
            target_size=(32, 32),
            class_mode='categorical',
            batch_size=256
        )

    def get_generators(self):
        return (self.train_generator, self.test_generator)

if __name__ == "__main__":
    train_df, test_df = DataManager().get_data()
    train_generator, test_generator = Generators(train_df, test_df).get_generators()

    for img_batch, target_batch in train_generator:
        print(img_batch.shape)
        break
