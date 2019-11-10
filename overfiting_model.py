import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

images_dir = "/storage/facial_expression_images/images"
full_df = pd.read_csv("/storage/affectnet+expw.csv")
large_subset, small_subset = train_test_split(full_df, test_size=0.1)
small_train_df, small_validation_df = train_test_split(small_subset, test_size=0.1)

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

# TODO : create training and validation generators, create a model that is likely to overfit, and train the model.

print(small_train_df.head(), small_validation_df.head())
