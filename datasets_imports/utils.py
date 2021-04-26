from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def extract_scale_format_data(X, y):
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()
    # Scale data to have mean 0 and variance 1
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split the data set into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.33, random_state=42)
    # size for the future
    n_features = X.shape[1]
    n_classes = Y.shape[1]
    return X_train, Y_train, X_test, Y_test, (n_features, n_classes)
