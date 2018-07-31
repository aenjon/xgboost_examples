#!/usr/bin/env python

import numpy as np
import xgboost as xgb

data = xgb.DMatrix(np.array([[1, 2, 3, 4]]))

# bst = xgb.Booster({'nthread': 4})  # init model
bst = xgb.Booster()
bst.load_model('model.bst')  # load data

result = bst.predict(data)
print(result)
