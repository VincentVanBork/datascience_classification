import csv

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from datasets_imports.iris import load_iris_for_neural

sizes_train, labels_train, sizes_test, labels_test, dims = load_iris_for_neural()
print("DIMENSIONS:", dims)

train_ds = tf.data.Dataset.from_tensors(
    (sizes_train, labels_train))

test_ds = tf.data.Dataset.from_tensors((sizes_test, labels_test))


class IrisModel(Model):
    def __init__(self, num_layers, size_nodes):
        super(IrisModel, self).__init__()
        self.num_layers = num_layers
        # self.input_layer = Dense(8, activation='relu', input_shape=(4,))
        for i in range(num_layers + 1):
            setattr(self, f'hidden{i}', Dense(size_nodes, activation='relu', input_shape=(4,)))
        # self.compact = Dense(4, activation='relu')
        self.out_layer = Dense(dims[1], activation="softmax")

    def call(self, x):
        # x = self.input_layer(x)
        for i in range(self.num_layers + 1):
            layer = getattr(self, f'hidden{i}')
            x = layer(x)
        # x = self.compact(x)
        return self.out_layer(x)

if __name__ == "__main__":
    for layers_diff in range(9):
        for nodes in [4 , 12,17,32]:
            model = IrisModel(num_layers=layers_diff, size_nodes=nodes)

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


            EPOCHS = 800

            with open(f'iris_model_{layers_diff}_n{nodes}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'loss', 'accuracy', 'test_loss', 'test_accuracy', "nodes_per_layer", "layers"])

                for epoch in range(EPOCHS):

                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    test_loss.reset_states()
                    test_accuracy.reset_states()
                    for input_values, output_values in train_ds:
                        train_step(input_values, output_values)

                    for input_values, output_values in test_ds:
                        test_step(input_values, output_values)

                    if epoch in [50, 100, 400, 200, 500, 600, 700, 799]:
                        writer.writerow([epoch + 1,
                                         f"{train_loss.result()}",
                                         f"{train_accuracy.result() * 100}",
                                         f"{test_loss.result()}",
                                         f"{test_accuracy.result() * 100}",
                                         f"{nodes}",
                                         layers_diff+1])
                    # print(
                    #     f'Epoch {epoch + 1}, '
                    #     f'Loss: {train_loss.result()}, '
                    #     f'Accuracy: {train_accuracy.result() * 100}, '
                    #     f'Test Loss: {test_loss.result()}, '
                    #     f'Test Accuracy: {test_accuracy.result() * 100}'
                    # )

            print(model.summary())
            # model.save(f"./iris_model{layers_diff}_0.33_test")
