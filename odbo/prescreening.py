import numpy as np
import warnings
from pyod.models.xgbod import xgbod

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



