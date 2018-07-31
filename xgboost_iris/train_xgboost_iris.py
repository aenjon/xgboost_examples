#!/usr/bin/env python

import pickle
from sklearn import datasets
import xgboost as xgb

from sklearn.externals import joblib

# Load the Iris dataset
iris = datasets.load_iris()

# Load data into DMatrix object
dtrain = xgb.DMatrix(iris.data, label=iris.target)

# Train XGBoost model
bst = xgb.train({}, dtrain, 20)

# Export the classifier to a file
bst.save_model('./model.bst')

joblib.dump(bst, 'model.joblib')

with open('model.pkl', 'wb') as model_file:
  pickle.dump(bst, model_file)
