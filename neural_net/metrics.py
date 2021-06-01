import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import tensorflow as tf
from neural_net.cancer_neural import CancerModel
from neural_net.inputs import cancer_neural_data_input, cancer_get_model_setup, \
    iris_data_input, prepare_iris_model, sensitivity, specificity, \
    mnist_data_input, prepare_mnist_model

from neural_net.iris_neural import IrisModel


def cancer_metrics():
    train_cancer, test_cancer = cancer_neural_data_input()

    model = cancer_get_model_setup(800, train_cancer, test_cancer)
    for input_tensor, output_values in test_cancer:
        predictions = model(input_tensor, training=False)
        # print(output_values)
    # print("___________________________")
    # print("PREDICTIONS: \n", predictions)

    rounded_predictions = np.argmax(predictions, axis=-1)
    labels_1d = np.argmax(output_values, axis=-1)
    # print(labels_1d, rounded_predictions)

    cancer_confusion_matrix = tf.math.confusion_matrix(
        labels_1d, rounded_predictions
    )
    confusion_matrix_object = confusion_matrix(labels_1d, rounded_predictions)
    print(confusion_matrix_object)
    print(cancer_confusion_matrix)

    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.update_state(labels_1d, rounded_predictions)
    print("ACCURACY", accuracy_metric.result().numpy())

    precision_metric = tf.keras.metrics.Precision()
    precision_metric.update_state(labels_1d, rounded_predictions)
    print("PRECISION", precision_metric.result().numpy())

    print("SENSITIVITY",
          sensitivity(output_values.numpy().astype(dtype="float64"),
                      predictions.numpy().astype(dtype="float64")))
    print("SPECIFITY",
          specificity(output_values.numpy().astype(dtype="float64"),
                      predictions.numpy().astype(dtype="float64")))


def iris_metrics():
    print("______________________")
    print("______________________")
    print("______________________")
    print("______________________")
    print("IRIS")
    train_iris, test_iris = iris_data_input()

    model = prepare_iris_model(400, train_iris, test_iris)

    for input_tensor, output_values in test_iris:
        predictions = model(input_tensor, training=False)

    rounded_predictions = np.argmax(predictions, axis=-1)
    labels_1d = np.argmax(output_values, axis=-1)

    iris_confusion_matrix = tf.math.confusion_matrix(
        labels_1d, rounded_predictions
    )
    confusion_matrix_object = confusion_matrix(labels_1d, rounded_predictions)
    print(confusion_matrix_object)
    print(iris_confusion_matrix)

    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.update_state(labels_1d, rounded_predictions)
    print("ACCURACY", accuracy_metric.result().numpy())

    # precision_metric = tf.keras.metrics.Precision()
    # print(labels_1d, rounded_predictions)
    #
    # precision_metric.update_state(labels_1d, rounded_predictions)
    # print("PRECISION", precision_metric.result().numpy())

    print("SENSITIVITY",
          sensitivity(output_values.numpy().astype(dtype="float64"),
                      predictions.numpy().astype(dtype="float64")))
    print("IRIS OUT", output_values)
    print("IRIS PRED",predictions)
    print("SPECIFITY",
          specificity(output_values.numpy().astype(dtype="float64"),
                      predictions.numpy().astype(dtype="float64")))

def mnist_metrics():
    print("______________________")
    print("______________________")
    print("______________________")
    print("______________________")
    print("MNIST")

    train_mnist, test_mnist = mnist_data_input()
    model = prepare_mnist_model(train_mnist, test_mnist)
    for input_tensor, output_values in test_mnist:
        predictions = model(input_tensor, training=False)
        # print(len(predictions))
    print(output_values)
    print(predictions)
    rounded_predictions = np.argmax(predictions, axis=-1)

    labels_1d = output_values.numpy()
    # print("LABLEST MNIST", labels_1d)
    # print("PREDICTIONS MNSIT", rounded_predictions)
    # mnist_confusion_matrix = tf.math.confusion_matrix(
    #     labels_1d, rounded_predictions
    # )
    confusion_matrix_object = confusion_matrix(labels_1d, rounded_predictions)
    print(confusion_matrix_object)
    # print(mnist_confusion_matrix)

    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.update_state(labels_1d, rounded_predictions)
    print("ACCURACY", accuracy_metric.result().numpy())

    # precision_metric = tf.keras.metrics.Precision()
    # print(labels_1d, rounded_predictions)
    #
    # precision_metric.update_state(labels_1d, rounded_predictions)
    # print("PRECISION", precision_metric.result().numpy())

    print("SENSITIVITY",
          sensitivity(output_values.numpy().astype(dtype="float64"),
                      rounded_predictions.astype(dtype="float64")))
    print("SPECIFITY",
          specificity(output_values.numpy().astype(dtype="float64"),
                      rounded_predictions.astype(dtype="float64")))


if __name__ == "__main__":
    ...
    # iris_metrics()
    # mnist_metrics()
