import csv

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from datasets_imports.cancer import load_cancer_for_neural

sizes_train, labels_train, sizes_test, labels_test, dims = load_cancer_for_neural()
print("DIMENSIONS:", dims)

train_ds = tf.data.Dataset.from_tensors(
    (sizes_train, labels_train))

test_ds = tf.data.Dataset.from_tensors((sizes_test, labels_test))


class CancerModel(Model):
    def __init__(self, num_layers):
        super(CancerModel, self).__init__()
        self.num_layers = num_layers
        # self.hidden0 = Dense(8, activation='relu', input_shape=(30,))
        # self.hidden1 = Dense(8, activation='relu', input_shape=(30,))
        # self.hidden2 = Dense(8, activation='relu', input_shape=(30,))
        # self.hidden3 = Dense(8, activation='relu', input_shape=(30,))
        for i in range(num_layers):
            setattr(self, f'hidden{i}', Dense(8, activation='relu', input_shape=(30,)))
        # self.compact = Dense(4, activation='relu')
        self.out_layer = Dense(dims[1], activation="softmax")

    def call(self, x):
        # x = self.input_layer(x)
        for i in range(self.num_layers):
            layer = getattr(self, f'hidden{i}')
            x = layer(x)
        # x = self.compact(x)
        return self.out_layer(x)

if __name__ == "__main__":
    for layers_diff in [1, 2, 3, 4, 5]:
        model = CancerModel(layers_diff)

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


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

        EPOCHS = 400

        with open(f'cancer_model_{layers_diff}_{EPOCHS}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'loss', 'accuracy', 'test_loss', 'test_accuracy'])

            for epoch in range(EPOCHS):

                train_loss.reset_states()
                train_accuracy.reset_states()
                test_loss.reset_states()
                test_accuracy.reset_states()
                for input_values, output_values in train_ds:
                    train_step(input_values, output_values)

                for input_values, output_values in test_ds:
                    test_step(input_values, output_values)

                writer.writerow([epoch + 1,
                                 f"{train_loss.result()}",
                                 f"{train_accuracy.result() * 100}",
                                 f"{test_loss.result()}",
                                 f"{test_accuracy.result() * 100}"])
                #
                # print(
                #     f'Epoch {epoch + 1}, '
                #     f'Loss: {train_loss.result()}, '
                #     f'Accuracy: {train_accuracy.result() * 100}, '
                #     f'Test Loss: {test_loss.result()}, '
                #     f'Test Accuracy: {test_accuracy.result() * 100}'
                # )

            print(model.summary())
            # model.save(f"./cancer_model{layers_diff}_0.33_test")
