import numpy as np
import warnings
from pyod.models.xgbod import xgbod



def sp_label(X, Y, thres=None, fraction=0.1):
   labels = np.zeros(len(Y_train))
   if thres == None:
       sort_ids = np.argsort(Y)
       labels[sort_ids[0:int(len(Y)*fraction)]] = np.ones(int(len(Y)*fraction))
   else:   
       labels = np.zeros(len(Y_train))
       outlier = [k for k, x in enumerate(Y) if x <= thres]
       labels[outlier] = np.ones(len(outlier))
   return labels 

class XGBOD(xgbod):
    def __init__(*args, **kwargs):
        super(XGBOD, self).__init__(*args, **kwargs)
        if 'estimator_list' not in kwargs:
            from pyod.models.knn import KNN
            from pyod.models.lof import LOF
            from pyod.models.ocsvm import OCSVM
            from pyod.models.iforest import IForest
            self.estimator_list = [KNN(), LOF(), OCSVM(), IForest()]


class LAMCTS():
    pass
