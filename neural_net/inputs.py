from datasets_imports.cancer import load_cancer_for_neural
import tensorflow as tf

from neural_net.cancer_neural import CancerModel
from datasets_imports.iris import load_iris_for_neural
from neural_net.iris_neural import IrisModel
from tensorflow.keras import backend as K

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(
        K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def cancer_neural_data_input():
    sizes_train, labels_train, sizes_test, labels_test, dims = load_cancer_for_neural()
    # print("DIMENSIONS:", sizes_train.shape)

    train_ds = tf.data.Dataset.from_tensors(
        (sizes_train, labels_train))

    test_ds = tf.data.Dataset.from_tensors((sizes_test, labels_test))
    # print("TEST_DS, SHAPE IS", test_ds)
    return train_ds, test_ds


def cancer_get_model_setup(epoch, train_ds, test_ds):
    model = CancerModel(num_layers=5)
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

    for epoch in range(epoch):

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for input_values, output_values in train_ds:
            train_step(input_values, output_values)

        for input_values, output_values in test_ds:
            test_step(input_values, output_values)
    return model


def iris_data_input():
    sizes_train, labels_train, sizes_test, labels_test, dims = load_iris_for_neural()
    print("DIMENSIONS:", dims)

    train_ds = tf.data.Dataset.from_tensors(
        (sizes_train, labels_train))

    test_ds = tf.data.Dataset.from_tensors((sizes_test, labels_test))
    return train_ds, test_ds


def prepare_iris_model(epoch, train_ds, test_ds):
    model = IrisModel(num_layers=4, size_nodes=8)
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

    for epoch in range(epoch):

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for input_values, output_values in train_ds:
            train_step(input_values, output_values)

        for input_values, output_values in test_ds:
            test_step(input_values, output_values)
    return model