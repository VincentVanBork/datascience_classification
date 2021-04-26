import numpy as np
import pandas as pd

import os

# Data understanding part
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')


print(train_data.shape)
print(test_data.shape)
print(train_data.head())
print(test_data.head())

# there are no missing values in the dataset
print(train_data.isnull().sum().head(10))
print(test_data.isnull().sum().head(10))
print(train_data.describe())
print(test_data.describe())

# Visualizing the stuff
order = list(np.sort(train_data['label'].unique()))
sns.countplot(train_data['label'])

## Visualizing the number of class and counts in the datasets
plt.plot(figure = (16,10))
g = sns.countplot( train_data["label"], palette = 'icefire')
plt.title('NUmber of digit classes')
train_data.label.astype('category').value_counts()

# Plotting some samples as well as converting into matrix

four = train_data.iloc[3, 1:]
four.shape
four = four.values.reshape(28,28)
plt.imshow(four, cmap='gray')
plt.title("Digit 4")

seven = train_data.iloc[6, 1:]
seven.shape
seven = seven.values.reshape(28, 28)
plt.imshow(seven, cmap='gray')
plt.title("Digit 7")

# average feature values
round(train_data.drop('label', axis=1).mean(), 2)

# Rescalling values

## Separating the X and Y variable

y = train_data['label']

## Dropping the variable 'label' from X variable 
X = train_data.drop(columns = 'label')

## Printing the size of data 
print(train_data.shape)

## Normalization

X = X/255.0
test_data = test_data/255.0

print("X:", X.shape)
print("test_data:", test_data.shape)

# scaling the features
from sklearn.preprocessing import scale
X_scaled = scale(X)

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


