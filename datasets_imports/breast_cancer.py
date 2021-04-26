from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_breast_cancer_for_neural():
    """
        x,y train and x,y test
    """
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    # TODO: Normalize the stuff
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)

    return X_train, X_test, y_train, y_test
 