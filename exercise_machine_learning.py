# -*- coding: utf-8 -*-
"""exercise_machine learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ARLPauNgAX3OhVGLWqb0agOqRjz4Sg4s
"""

##create a simple dataset of people

data = {'Name': ["John", "Anna", "Peter", "Linda"],
 'Location' : ["New York", "Paris", "Berlin", "London"],
 'Age' : [24, 13, 53, 33]
 }

import pandas as pd

data_pandas =  pd.DataFrame(data)
data_pandas

data_pandas[data_pandas.Age> 14]

data_pandas.Age, type(data_pandas.Age)

import sys

sys.version

!pip install mglearn

import mglearn


X, y = mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

X.shape

! pip install matplotlib

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], y)
plt.ylim(0, 10)
plt.xlim(0, 10)
plt.xlabel("first feature")
plt.ylabel("second feature")

X, X[:, 0] , X[:, 1]

x, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(x, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")

x.shape

###breast cancer prediction model

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

dir(cancer)

cancer.keys()

cancer["DESCR"]

cancer["feature_names"], cancer["target_names"]

type(cancer)

X = cancer.data
y = cancer.target

X, y

X.shape, y.shape

plt.scatter(X[:, 0], X[:, 1], y)

plt.plot(X, y)

mglearn.plots.plot_knn_classification(n_neighbors=5)

from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape, X_test.shape

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

clf

prediction = clf.predict(X_test)

prediction

import numpy as np


np.mean(prediction==y_test)

clf.score(X_test, y_test)

### breast cancer for the knn


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier


knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

score = knn_model.score(X_test, y_test)

score

for neighbors in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=neighbors)
    knn_model.fit(X_train, y_train)
    score = knn_model.score(X_test, y_test)
    print(score)

X_train.shape, X_test.shape

from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg_model = KNeighborsRegressor(n_neighbors=3)

reg_model.fit(X_train, y_train)

reg_model.score(X_test, y_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
 # make predictions using 1, 3, or 9 neighbors
 reg = KNeighborsRegressor(n_neighbors=n_neighbors)
 reg.fit(X_train, y_train)
 ax.plot(line, reg.predict(line))
 ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
 ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
 ax.set_title(
 "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
 n_neighbors, reg.score(X_train, y_train),
 reg.score(X_test, y_test)))
 ax.set_xlabel("Feature")
 ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
 "Test data/target"], loc="best")

## linear plot for the linear models
import mglearn

mglearn.plots.plot_linear_regression_wave()

### linear regression examples

from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

lr.score(X_test, y_test)

prediction = lr.predict(X_test)

prediction

lr.coef_, lr.intercept_

### linear regression for the boston housing

from sklearn.datasets import fetch_california_housing

boston = fetch_california_housing()

X = boston.data[:500]
y = boston.target[:500]

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)


lr.score(X_train, y_train), lr.score(X_test, y_test)

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)

ridge.score(X_train, y_train), ridge.score(X_test, y_test)
#

### less restricted and more related to the linear regression

ridge = Ridge(0.1).fit(X_train, y_train)

ridge.score(X_train, y_train), ridge.score(X_test, y_test)

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)

lasso.score(X_train, y_train), lasso.score(X_test, y_test)

lasso.coef_

lasso = Lasso(0.01).fit(X_train, y_train)

score = lasso.score(X_train, y_train), lasso.score(X_test, y_test)

lasso.coef_, score

### linear classification

from sklearn.linear_model import LogisticRegression

X, y = mglearn.datasets.make_forge()

from sklearn.svm import LinearSVC

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
  clf = model.fit(X, y)
  mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
  ax=ax, alpha=.7)
  mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
  ax.set_title("{}".format(clf.__class__.__name__))
  ax.set_xlabel("Feature 0")
  ax.set_ylabel("Feature 1")

axes[0].legend()

mglearn.plots.plot_linear_svc_regularization()

#### breast cancer logostic regression

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegression().fit(X_train, y_train)

logreg.score(X_train, y_train), logreg.score(X_test, y_test)

import matplotlib.pyplot as plt

X.shape

plt.scatter(X[:, 0], X[:, 1], y)

for c, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
  lr_l1 = LogisticRegression(C=c, penalty="l1", solver='liblinear').fit(X_train, y_train)
  print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
  c, lr_l1.score(X_train, y_train)))
  print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
  c, lr_l1.score(X_test, y_test)))
  plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(c))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")



from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)

X.shape, y.shape

import matplotlib.pyplot as plt
import numpy as np

# Assuming X contains your features and y is the class labels
# Get the unique classes in y and define a list of markers
classes = np.unique(y)
markers = ['o', 's', 'D', '^', 'v']  # Example marker styles; add more if needed

# Plot each class with a separate color, label, and marker
for i, class_value in enumerate(classes):
    plt.scatter(
        X[y == class_value, 0],
        X[y == class_value, 1],
        label=f'Class {class_value}',
        marker=markers[i % len(markers)],  # Cycle through markers if classes exceed marker count
        alpha=0.7,
        edgecolor='k'
    )

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title="Classes")
plt.show()

### train a linear svc

linear_svc = LinearSVC().fit(X, y)

linear_svc.coef_, linear_svc.intercept_, X.shape

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
print(line)
for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_,
                                  ['b', 'r', 'g']):
  plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

