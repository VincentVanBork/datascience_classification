from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

from datasets_imports.utils import extract_scale_format_data


def load_cancer_for_neural():
    """
        x,y train and x,y test,  (in, out) <- dim returned
    """

    cancer = load_breast_cancer()
    X = cancer['data']
    y = cancer['target']
    print(cancer)
    # One hot encoding
    return extract_scale_format_data(X, y)

