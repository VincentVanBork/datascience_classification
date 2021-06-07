import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve

from plot_learning_curve import plot_learning_curve

def experiment(svc):
    # Plot learning-curve
    plot_learning_curve(svc, "some title", X_train, y_train)
    plt.savefig('learning_curve.png')

    model_linear.fit(X_train, y_train)

    # predict
    y_pred = model_linear.predict(X_test)

    # Confussion matrix
    plot_confusion_matrix(svc, X_test, y_test)
    plt.savefig('confussion_matrix' + '.png')

    # Prescision, accuracy, sensitivity and specifity
    print("report:", metrics.classification_report(y_true=y_test, y_pred=y_pred), "\n")

    metrics.plot_roc_curve(svc, X_test, y_test)
    plt.savefig('roc.png')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# commented out task 5 stuff
# model_linear = SVC(kernel='linear')
# experiment(model_linear)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)

# confusion matrix and accuracy

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Linear
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

# Optimal stuff RBF
model = SVC(C=10, kernel="rbf")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

# Random stuff
model = SVC(C=5, kernel="rbf")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
