import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve

def experiment(svc):
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

# TODO: Czy trzeba przeprowadzić normalizację?
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)

# linear model

model_linear = SVC(kernel='linear')
experiment(model_linear)