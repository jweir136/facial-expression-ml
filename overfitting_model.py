import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self):
        full_df = pd.read_csv("/storage/affectnet+expw.csv")
        large_df, small_df = train_test_split(full_df, test_size=0.1)
        self.train_df, self.test_df = train_test_split(full_df, test_size=0.25)
    def get_data(self):
        return (self.train_df, self.test_df)

class Generators:
    def __init__(self, train_df, test_df):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255.,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            
        )
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
        self.train_generator = train_datagen.flow_from_dataframe(
            directory = "/storage/facial_expression_images/images",
            dataframe = train_df,
            x_col="filenames",
            y_col="expression",
            target_size=(32, 32),
            class_mode='categorical',
            batch_size=256
        )
        self.test_generator = test_datagen.flow_from_dataframe(
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
        self.earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        filepath = "/storage/facial_expression_weights/overfitting_model/saved-facial-expression-model-1-{epoch:02d}-{val_acc:.2f}.hdf5"
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
        self.save_best = tf.keras.callbacks.ModelCheckpoint('/storage/facial_expression_weights/overfitting_model/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    def get_early_stopping(self):
        return self.earlyStopping
    
    def get_checkpoint(self):
        return self.checkpoint
    
    def get_save_best(self):
        return self.save_best

class OverfittingModel:
    def __init__(self):
        input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(input_layer)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(4, activation='softmax')(x)

        self.model = tf.keras.models.Model(input_layer, x)
        
    def get_model(self):
        return self.model

if __name__ == "__main__":
    train_df, test_df = DataManager().get_data()
    train_generator, test_generator = Generators(train_df, test_df).get_generators()
    print("GOT DATA AND GENERATORS")

    model = OverfittingModel().get_model()
    print("LOADED MODEL")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print("COMPILIED MODEL")
    print("STARTING TRAINING")
    history = model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=test_generator,
        callbacks=[CallbackManager().get_early_stopping(), CallbackManager().get_checkpoint(), CallbackManager().get_save_best()],
        verbose=2
    )
