import numpy as np
import warnings

class FeatureTransform(object):
   def __init__(self, raw_vars, cat_features=None, Y=None, categories=None):
     self._raw_vars = raw_vars
     self._cat_features = cat_features
     self._Y = Y
     self._catergories = categories


class AvgMeasurement(FeatureTransform):
   def __init__(self, raw_vars, y, cat_features=None, categories=None):
     super(AvgMeasurement, self).__init__(raw_vars=raw_vars, cat_features=cat_features, y=y, categories=categories)
     self._name = 'AvgMeasurement'
     if categories == None:
        categories = []
        for i in range(raw_vars.shape[1]):
           choice = np.array(set(raw_vars[:, i]))
           categories.append(choice)
     self._categories = categories
     if cat_features == None:
        cat_features = []
        for i in range(raw_vars.shape[1]):
           feature_choice = np.empty(len(categories[i]))
           for j in range(len(categories[i])):
              ids = [k for k, x in enumerate(raw_vars[:,i]) if x == categories[i][j]]
              feature_choice[j] = np.mean(Y[ids])
           cat_features.append(feature_choice)
     self._cat_features = cat_features

   def transform(self, raw_vars):
     transformed_feature = np.ones(raw_vars.shape)* np.nan
     for i in range(raw_vars.shape[1]):
        for j in range(len(self._categories[i])):
          ids = [k for k, x in enumerate(raw_vars[:,i]) if x == self._categories[i][j]]
          transform_feature[ids, i] = self._cat_features[i][j]
          try:
             np.isnan(transformed_feature.sum())
          except InputError as err:
             print('InputError: A wrong experimental variable at ', np.argwhere(np.isnan(transformed_feature))) 
     return transfomed_feature
 
