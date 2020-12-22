# Tony Amin
# Robert Deluna
# CS 489 - Assignment 6

# This program implements a muli-layer perceptron (MLP) regressor to solve a
# regression problem.

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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# This implements a multi-layer percetron algorithm that trains using backpropogation
from sklearn.neural_network import MLPRegressor

# load data
data = pd.read_csv("haberman.csv")
data.head()

X = data[["age", "operation_year", "ax_nodes_detected"]]
Y = data.survival_stat

# divide data into training, and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, shuffle = False, stratify = None, )

# adding the bias vector
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# does our data need to be standardized?
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# hidden layer is by default = 100
# 5 hidden layers with 3 nodes in each hidden layer
# linear activtion function 'relu'
print('')
print('MLP Classifier: 5 layers, 3 nodes, 500 iterations, rectified linear unit "relu"')
print('-------------------------------------------------------------------------------')
regClass1 = MLPRegressor(max_iter = 500, hidden_layer_sizes = (5,3), activation = 'relu', learning_rate = 'constant', solver =  'lbfgs' )
regClass1.fit(X_train,Y_train)
pred1 = regClass1.predict(X_test)

print(classification_report(Y_test, pred1, zero_division = 1))
print(confusion_matrix(Y_test, pred1))
print("class 1 iterations: ", mlpClass1.n_iter_)

# # 3 hidden layers with 11 nodes in each hidden layer
# # linear activtion function 'logistic'
# print('')
# print('MLP Classifier: 3 layers, 11 nodes, 500 iterations, logistic activation function (Test Dataset)')
# print('--------------------------------------------------------------------------------')
# mlpClass2 = MLPClassifier(max_iter = 500, hidden_layer_sizes = (3,11), activation = 'logistic', learning_rate = 'constant', solver =  'sgd' )
# mlpClass2.fit(X_train,Y_train)
# pred2 = mlpClass2.predict(X_test)
#
# print(classification_report(Y_test, pred2, zero_division = 1))
# print(confusion_matrix(Y_test, pred2))
# print("class 2 iterations: ", mlpClass2.n_iter_)
#
# # 5 hidden layers, 11 neurons in each hidden layer
# # using a logistic sigmoid function as the activation function
# print('')
# print('MLP Classifier: 3 layers, 11 nodes, 750 iterations, hyperbolic tangent function (Test Dataset)')
# print('-------------------------------------------------------------------------------')
# mlpClass3 = MLPClassifier(max_iter = 750, hidden_layer_sizes = (5,11), activation = 'tanh', learning_rate = 'constant', solver = 'adam' )
# mlpClass3.fit(X_train,Y_train)
# pred3 = mlpClass3.predict(X_test)
#
# print(classification_report(Y_test, pred3, zero_division = 1))
# print(confusion_matrix(Y_test, pred3))
# print("class 3 iterations: ", mlpClass3.n_iter_)
#
# # 100 hidden layers, 50 nodes in each layer
# # using logistic sigmod function as activation function
# print('')
# print('MLP Classifier: 50 layers, 15 nodes, 1000 iterations, hyperbolic tangent function (Test Dataset)')
# print('---------------------------------------------------------------------------------')
# mlpClass4 = MLPClassifier(max_iter = 1000, hidden_layer_sizes = (50,15), activation = 'logistic', learning_rate = 'constant', solver = 'adam' )
# mlpClass4.fit(X_train,Y_train)
# pred4 = mlpClass4.predict(X_test)
#
# print(classification_report(Y_test, pred4, zero_division = 1))
# print(confusion_matrix(Y_test, pred4))
# print("class 4 iterations: ", mlpClass4.n_iter_)
#
# # start train dataset
#
# # 3 hidden layers with 11 nodes in each hidden layer
# # linear activtion function 'logistic'
# print('')
# print('MLP Classifier: 3 layers, 11 nodes, 500 iterations, logistic activation function (Train Dataset)')
# print('--------------------------------------------------------------------------------')
# mlpClass2 = MLPClassifier(max_iter = 500, hidden_layer_sizes = (3,11), activation = 'logistic', learning_rate = 'constant', solver =  'sgd' )
# mlpClass2.fit(X_train,Y_train)
# pred2 = mlpClass2.predict(X_train)
#
# print(classification_report(Y_train, pred2, zero_division = 1))
# print(confusion_matrix(Y_train, pred2))
# print("class 2 iterations: ", mlpClass2.n_iter_)
#
# # 5 hidden layers, 11 neurons in each hidden layer
# # using a logistic sigmoid function as the activation function
# print('')
# print('MLP Classifier: 3 layers, 11 nodes, 750 iterations, hyperbolic tangent function (Train Dataset)')
# print('-------------------------------------------------------------------------------')
# mlpClass3 = MLPClassifier(max_iter = 750, hidden_layer_sizes = (5,11), activation = 'tanh', learning_rate = 'constant', solver = 'adam' )
# mlpClass3.fit(X_train,Y_train)
# pred3 = mlpClass3.predict(X_train)
#
# print(classification_report(Y_train, pred3, zero_division = 1))
# print(confusion_matrix(Y_train, pred3))
# print("class 3 iterations: ", mlpClass3.n_iter_)
#
# # 100 hidden layers, 50 nodes in each layer
# # using logistic sigmod function as activation function
# print('')
# print('MLP Classifier: 50 layers, 15 nodes, 1000 iterations, hyperbolic tangent function (Train Dataset)')
# print('---------------------------------------------------------------------------------')
# mlpClass4 = MLPClassifier(max_iter = 1000, hidden_layer_sizes = (50,15), activation = 'logistic', learning_rate = 'constant', solver = 'adam' )
# mlpClass4.fit(X_train,Y_train)
# pred4 = mlpClass4.predict(X_train)
#
# print(classification_report(Y_train, pred4, zero_division = 1))
# print(confusion_matrix(Y_train, pred4))
# print("class 4 iterations: ", mlpClass4.n_iter_)
