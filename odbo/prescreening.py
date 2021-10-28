import numpy as np
import warnings
from pyod.models.xgbod import XGBOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD


def XGBOD(
        X,
        y,
        estimator_list=[KNN(), LOF(), OCSVM(),
                        IForest()],
        eval_metric='error',
        random_state=0):
    model = XGBOD(
        estimator_list=estimator_list,
        eval_metric=eval_metric,
        random_state=random_state)
    model.fit(X, y)
    return model

def LAMCTS():
    pass



