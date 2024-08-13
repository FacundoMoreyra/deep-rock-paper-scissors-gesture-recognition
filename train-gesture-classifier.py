import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load the dataset
X = np.load('rps_dataset.npy') # hand data
y = np.load('rps_labels.npy') # label (rock, paper, scissor.)

# One-hot encoding
y = tf.keras.utils.to_categorical(y, 3)

# Define the model
model = models.Sequential([
    layers.Input(shape=(42,)),    # 42 inputs (21 dots with x and y)
    layers.Dense(64, activation='relu'), # Dense layer with activation relu
    layers.Dense(32, activation='relu'), # Dense layer with activation relu
    layers.Dense(3, activation='softmax')  # 3 output classes (rock, paper, scissors)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, batch_size=32)

# Save the model
model.save('rps_model.h5')
