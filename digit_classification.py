# -*- coding: utf-8 -*-
"""Digit Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/141pj8o8KUJ0ZGD5uEv1kb3UKaO3G5FUU

I will be using many different ML models to classify digits
"""

import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
digits = load_digits()

"""Here I'll do some exploratory analysis of the dataset. This analysis is exactly the same as what I did for the SVM exercise"""

dir(digits)

digits.target

digits.target_names

df = pd.DataFrame(digits.data, digits.target)
df.head()

df['Target'] = digits.target
df.head()

"""Now I'll create the training and testing datasets for each model to work with."""

from sklearn.model_selection import train_test_split
x = df.drop(['Target'], axis = 'columns')
y = df.Target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

"""The first model I'll be using is logistic classification. I'll use GridSearch to find the optimal parameters."""

from sklearn.linear_model import LogisticRegression
logiClf = LogisticRegression(max_iter=10000)
logiClf.fit(x_train, y_train)
logiClf.score(x_test, y_test)

logiPred = logiClf.predict(x_test)
logiCM = confusion_matrix(logiPred, y_test)
plt.figure(figsize=(10, 7))
sns.heatmap(logiCM, annot=True)

"""Second, I'll use a SVM"""

from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
svm.score(x_test, y_test)

svmPred = svm.predict(x_test)
svmCM = confusion_matrix(logiPred, y_test)
plt.figure(figsize=(10, 7))
sns.heatmap(svmCM, annot=True)

"""Third, a random forest"""

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc.score(x_test, y_test)

rfcPred = rfc.predict(x_test)
rfcCM = confusion_matrix(rfcPred, y_test)
plt.figure(figsize=(10, 7))
sns.heatmap(rfcCM, annot=True)

"""Finally, I'll use a neural network. We didn't cover this in class but I want to research it."""

from sklearn.neural_network import MLPClassifier
nn = MLPClassifier()
nn.fit(x_train, y_train)
nn.score(x_test, y_test)

nnPred = nn.predict(x_test)
nnCM = confusion_matrix(nnPred, y_test)
plt.figure(figsize=(10, 7))
sns.heatmap(nnCM, annot=True)