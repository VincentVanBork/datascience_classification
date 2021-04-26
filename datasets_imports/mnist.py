import tensorflow as tf


def load_mnist_for_neural():
    """
        x,y train and x,y test returned
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #color rgb is 255
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #unpack with new axis
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    return x_train, y_train, x_test, y_test