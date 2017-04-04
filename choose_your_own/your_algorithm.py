#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1]
              for ii in range(0, len(features_train)) if labels_train[ii] == 1]


# initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
##########################################################################


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sys
from sklearn.grid_search import GridSearchCV

# AdaBoost: {'n_estimators': 100, 'base_estimator__criterion': 'entropy', 'algorithm': 'SAMME', 'base_estimator__splitter': 'random'}
#In [147]: %run your_algorithm.py
#0.940036672587
#0.952
param_grid = {
    "base_estimator__criterion": ["entropy", "gini"],
    "base_estimator__splitter":   ["best", "random"],
    "base_estimator__max_depth": [None, 1, 2],
    "n_estimators": [1, 10, 50, 100, 150],
    "algorithm": ["SAMME", "SAMME.R"],
    "random_state": [1, 5, 9, 10, 20, 30 , 50]
    #   "learning_rate": [0.5,0.9,1,1.1,1.5]
}

# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
# clf = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', n_jobs=4)

# n = int(sys.argv[1])
clf = AdaBoostClassifier(DecisionTreeClassifier(
    max_depth=1, criterion="entropy", splitter="random"), n_estimators=100, algorithm='SAMME', random_state=9)
# clf = RandomForestClassifier(n_estimators=n, max_depth=None,
#                        min_samples_split=100)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# print clf.best_params_

score = cross_val_score(clf, features_train, labels_train)
print score.mean()

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)


try:
  prettyPicture(clf, features_test, labels_test)
except NameError:
  pass
