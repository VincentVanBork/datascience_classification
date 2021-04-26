import tensorflow as tf
from keras.layers import Flatten
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from datasets_imports.iris import load_iris_for_neural
import numpy as np

sizes_train, labels_train, sizes_test, labels_test, dims = load_iris_for_neural()
print("DIMENSIONS:", dims)

train_ds = tf.data.Dataset.from_tensors(
    (sizes_train, labels_train))

test_ds = tf.data.Dataset.from_tensors((sizes_test, labels_test))


class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.input_layer = Dense(10, activation='relu', input_shape=(4,))
        # self.hidden = Dense(10, activation='relu')
        self.out_layer = Dense(dims[1], activation="softmax")

    def call(self, x):
        x = self.input_layer(x)
        # x = self.hidden(x)
        return self.out_layer(x)


model = IrisModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(sizes, labels):
    with tf.GradientTape() as tape:
        # recording gradients
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(sizes, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(sizes, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(sizes, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
#
# #
if __name__ == "__main__":
    EPOCHS = 5
    for epoch in range(EPOCHS):

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for input_values, output_values in train_ds:
            print("INPUTS", input_values, "OUTPUTS", output_values)

            train_step(input_values, output_values)

        # for input_values, output_values in test_ds:
        #     test_step(input_values, output_values)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
        model.save("./iris_model")
