"""
A simple regression analysis on the Boston housing data
========================================================

Here we perform a simple regression analysis on the Boston housing
data, exploring two types of regressors.

taken from: https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_boston_prediction.html

"""

from sklearn.datasets import load_boston

data = load_boston()

##############################################################################
# Simple prediction

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)


##############################################################################
# Prediction with gradient boosted tree

from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
expected = y_test

import pickle

pickle.dump(clf, open("model.pk", "wb"))
