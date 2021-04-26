import numpy as np
import pandas as pd

import os
# TODO: przepisz to 
print(os.listdir('../input'))

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

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
