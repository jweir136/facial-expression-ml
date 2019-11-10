import keras
import numpy as np
import matplotlib.pyplot as plt

generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
train_generator = generator.flow_from_dataframe(
    directory="/storage/facial_expression_images/images",
    dataframe="/storage/affectnet+expw.csv",
    x_col='filenames',
    y_col='expression',
    target_size=(32, 32),
    color_mode='grayscale',
    class_mode='categorical',
)
validation_generator = generator.flow_from_dataframe(
    directory="/storage/facial_expression_images/images",
    dataframe="/storage/affectnet+expw.csv",
    x_col="filenames",
    y_col="expression",
    target_size=(32, 32),
    color_mode='grayscale',
    class_mode='categorical',
)

for img_batch, target_batch in train_generator:
    print(target_batch[0])
    print(target_batch.shape)
    break
