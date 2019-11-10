import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import *

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

class CallbackManager:
    def __init__(self):
        self.earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')

    def get_early_stopping(self):
        return self.earlyStopping

class OverfittingModel:
    def __init__(self):
        input_layer = Input(shape=(32, 32, 3))
        x = Conv2D(64, (3, 3), activation='relu')(input_layer)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.0)(x)

        y = Conv2D(64, (3, 3), activation='relu')(input_layer)
        y = Conv2D(64, (3, 3), activation='relu')(y)
        y = Conv2D(64, (3, 3), activation='relu')(y)
        y = MaxPooling2D((2, 2))(y)
        y = Conv2D(32, (3, 3), activation='relu')(y)
        y = Conv2D(32, (3, 3), activation='relu')(y)
        y = Conv2D(32, (3, 3), activation='relu')(y)
        y = MaxPooling2D((2, 2))(y)
        y = Flatten()(y)
        y = Dropout(0.0)(y)

        concat = Concatenate([x, y])

        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(4, activation='softmax')(x)

        self.model = keras.models.Model(input_layer, x)
        
    def get_model(self):
        return self.model

if __name__ == "__main__":
    train_df, test_df = DataManager().get_data()
    train_generator, test_generator = Generators(train_df, test_df).get_generators()

    model = OverfittingModel().get_model()
    model.compile(optimizer='adam', loss='categorical', metrics=['acc'])
    history = model.fit_generator(
        train_generator,
        epochs=20,
        validation_data=test_generator,
        callbacks=[CallbackManager().get_early_stopping()]
    )

    X = range(len(history.history['val_acc']))
    plt.plot(X, history.history['acc'], 'r')
    plt.plot(X, history.history['val_acc'], 'bo')
    plt.savefig("/artifacts/training_history.png")
