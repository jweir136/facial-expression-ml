import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

images_dir = "/storage/facial_expression_images/images"
full_df = pd.read_csv("/storage/affectnet+expw.csv")
large_subset, small_subset = train_test_split(full_df, test_size=0.01)
small_train_df, small_validation_df = train_test_split(small_subset, test_size=0.1)

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
train_generator = datagen.flow_from_dataframe(
    dataframe=small_train_df,
    directory=images_dir,
    x_col="filenames",
    y_col="expression",
    batch_size=128,
    target_size=(96, 96)
)
validation_generator = datagen.flow_from_dataframe(
    dataframe=small_validation_df,
    directory=images_dir,
    x_col="filenames",
    y_col="expression",
    batch_size=128,
    target_size=(96, 96)
)

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

model = keras.models.Sequential([
    keras.layers.Conv2D(128, (3, 3), input_shape=(96, 96, 3), activation='relu'),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Conv2D(128, (3, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.Conv2D(32, (3, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Activation('relu'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.0),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit_generator(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    verbose=2,
    callbacks=[earlyStopping]
)

plt.plot(range(len(history.history['acc'])), history.history['acc'], 'r')
plt.plot(range(len(history.history['val_acc'])), history.history['val_acc'], 'bo')
plt.savefig("/artifacts/overfitting_training_history.png")
