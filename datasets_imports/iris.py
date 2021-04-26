import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datasets_imports.utils import extract_scale_format_data


def load_iris_for_neural():
    """
        x,y train and x,y test,  (in, out) <- dim returned
    """
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']
    # One hot encoding
    return extract_scale_format_data(X, y)



