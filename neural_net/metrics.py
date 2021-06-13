import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve
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
    print("_____________ ROCC _________")
    print(labels_1d)
    print(rounded_predictions)
    fpr, tpr, thresholds = roc_curve(labels_1d, rounded_predictions)

    # plot the roc curve for the model
    fig_roc_cancer, ax_roc_cancer = plt.subplots()
    fig_roc_cancer.suptitle("ROC dla zbioru cancer")
    ax_roc_cancer.plot(fpr, tpr, linestyle='--', label='No Skill')
    plt.show()


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
    # pri
    print("SPECIFITY",
          specificity(output_values.numpy().astype(dtype="float64"),
                      predictions.numpy().astype(dtype="float64")))
    print("_____________ ROCC _________")
    print(labels_1d)
    print(rounded_predictions)
    fig_roc_iris, ax_roc_iris = plt.subplots()
    fig_roc_iris.suptitle("ROC dla zbioru iris")

    fpr, tpr, thresholds = roc_curve(
        [1 if label == 0 else 0 for label in labels_1d],
        [1 if label == 0 else 0 for label in rounded_predictions])
    ax_roc_iris.plot(fpr, tpr, linestyle='--', label='1')

    fpr, tpr, thresholds = roc_curve(
        [1 if label == 1 else 0 for label in labels_1d],
        [1 if label == 1 else 0 for label in rounded_predictions])
    ax_roc_iris.plot(fpr, tpr, linestyle='--', label='2')
    fpr, tpr, thresholds = roc_curve(
        [1 if label == 2 else 0 for label in labels_1d],
        [1 if label == 2 else 0 for label in rounded_predictions])
    ax_roc_iris.plot(fpr, tpr, linestyle='--', label='3')

    plt.show()


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
    # print(output_values)
    # print(predictions)
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

    print("_____________ ROCC _________")
    print(labels_1d)
    print(rounded_predictions)
    # plot the roc curve for the model
    fig_roc_mnist, ax_roc_mnist = plt.subplots()
    fig_roc_mnist.suptitle("ROC dla zbioru mnist")
    for i in range(10):
        if i == 0:
            fpr, tpr, thresholds = roc_curve(
                [1 if label == 0 else 0 for label in labels_1d],
                [1 if label == 0 else 0 for label in rounded_predictions])
            ax_roc_mnist.plot(fpr, tpr, linestyle='--', label='1')
        else:
            fpr, tpr, thresholds = roc_curve(
                [1 if label == i else 0 for label in labels_1d],
                [1 if label == i else 0 for label in rounded_predictions])
            print(max(fpr), max(tpr))
            ax_roc_mnist.plot(fpr, tpr, linestyle='--', label='1')
    plt.show()

if __name__ == "__main__":
    ...
    # cancer_metrics()
    # iris_metrics()
    mnist_metrics()
