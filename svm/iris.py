from task5.utils import experiment
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)

# confusion matrix and accuracy

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
# print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

experiment(SVC(C=0.001, kernel='linear'), X_train, y_train, X_test, y_test)
experiment(SVC(C=0.01, kernel='linear'), X_train, y_train, X_test, y_test)
experiment(SVC(C=0.1, kernel='linear'), X_train, y_train, X_test, y_test)
experiment(SVC(C=1, kernel='linear'), X_train, y_train, X_test, y_test)
experiment(SVC(C=10, kernel='linear'), X_train, y_train, X_test, y_test)
experiment(SVC(C=100, kernel='linear'), X_train, y_train, X_test, y_test)

experiment(SVC(C=0.001, kernel='rbf'), X_train, y_train, X_test, y_test)
experiment(SVC(C=0.01, kernel='rbf'), X_train, y_train, X_test, y_test)
experiment(SVC(C=0.1, kernel='rbf'), X_train, y_train, X_test, y_test)
experiment(SVC(C=1, kernel='rbf'), X_train, y_train, X_test, y_test)
experiment(SVC(C=10, kernel='rbf'), X_train, y_train, X_test, y_test)
experiment(SVC(C=100, kernel='rbf'), X_train, y_train, X_test, y_test)

experiment(SVC(C=0.001, kernel='sigmoid'), X_train, y_train, X_test, y_test)
experiment(SVC(C=0.01, kernel='sigmoid'), X_train, y_train, X_test, y_test)
experiment(SVC(C=0.1, kernel='sigmoid'), X_train, y_train, X_test, y_test)
experiment(SVC(C=1, kernel='sigmoid'), X_train, y_train, X_test, y_test)
experiment(SVC(C=10, kernel='sigmoid'), X_train, y_train, X_test, y_test)
experiment(SVC(C=100, kernel='sigmoid'), X_train, y_train, X_test, y_test)