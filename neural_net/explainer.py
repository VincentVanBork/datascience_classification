import tensorflow as tf
from tensorflow import keras

from datasets_imports.iris import load_iris_for_neural
import numpy as np

model: tf.keras.Model = keras.models.load_model('./iris_model3_0.33_test')
# model.compile(optimizer='adam', loss=)

X_train, y_train, X_test, y_test, dims = load_iris_for_neural()
print("DIMENSIONS:", dims)

print(model)
train_ds = tf.data.Dataset.from_tensors(
    (X_train, y_train))

test_ds = tf.data.Dataset.from_tensors((X_test, y_test))
