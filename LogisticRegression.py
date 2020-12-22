# Implementing thelogisticregression with gradient descent algorithm to solve a binary classification problem.

# Tony Amin
# CS 489 - Assignment 3

# Dataset : Haberman's Survival Data - conducted between
#   1958 and 1970 at the University of Chicago's Billings Hospital on
#   the survival of patients who had undergone surgery for breast
#   cancer.

#  Number of Instances: 306
#  Number of Attributes: 4 (including the class attribute)
#
#  Information:
#   1. Age of patient at time of operation (numerical)
#   2. Patient's year of operation (year - 1900, numerical)
#   3. Number of positive axillary nodes detected (numerical)
#   4. Survival status (class attribute)
#         1 = the patient survived 5 years or longer
#         2 = the patient died within 5 year

import numpy as np
import pandas as pd
import statsmodels.api as sm


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def accuracy (Y_true, Y_pred):
    accuracy = np.sum(Y_true == Y_pred) / len(Y_true)
    return accuracy

# load data
data = pd.read_csv("haberman.csv")
data.head()

X = data[["age", "operation_year", "ax_nodes_detected"]]
Y = data.survival_stat

# divide data into training, and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, shuffle = False, stratify = None)

# adding the bias vector
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log',eta0=1, learning_rate="constant", max_iter=1000, tol=1e-10))
clf.fit(X_train, Y_train)
#print('clf: ', clf)
print('here: 40, 65, 0: ', clf.predict([[1, 40, 65, 0]]))


trainPredict = clf.predict(X_train)

# Training Dataset outputs evaluation metrics
print("\n")
print("Training Dataset")
print("----------------")
print("w: ", clf['sgdclassifier'].coef_)
print("Mean Sqaured Error: ", mean_squared_error(Y_train, trainPredict))
print("LR Classification accuracy: ", accuracy(Y_train, trainPredict))
print("LR Classification error: ", 1-accuracy(Y_train, trainPredict))
print("Learning Rate: ", 'constant')
print ("Score: ", clf.score(X_train,Y_train))

testPredict = clf.predict(X_test)

# Testing Dataset evaluation metrics
print("\n")
print("Testing Dataset")
print("---------------")
print("w: ", clf['sgdclassifier'].coef_)
print("Mean Sqaured Error: ", mean_squared_error(Y_test, testPredict))
print("LR Classification accuracy: ", accuracy(Y_test, testPredict))
print("LR Classification error: ", 1-accuracy(Y_test, testPredict))
print("Learning Rate: ", 'constant')
print ("Score: ", clf.score(X_train,Y_train))


