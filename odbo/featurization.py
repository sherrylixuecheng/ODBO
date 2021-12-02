import numpy as np
import warnings


class FeatureTransform(object):
    """Abstract base class for any FeatureTransformation method
    """

    def __init__(self,
                 raw_vars,
                 cat_features=None,
                 Y=None,
                 categories=None,
                 mode='independent'):
        """Constructor for the FeatureTransform base class.
        Args:
            raw_vars : Input experiments expressed using raw variable names
            cat_features : Featurizations of different categories of each variable
            Y : Input training measurements
            categories : Categories/Choices of each variable
        """
        self._raw_vars = raw_vars
        self._cat_features = cat_features
        self._Y = Y
        self._catergories = categories
        self._mode = mode

    def transform(self, raw_vars):
        """Transform input experiments to standard encodings
        Args:
            raw_vars : Input experiments expressed using raw variable names        
        """
        transformed_feature = np.ones(raw_vars.shape) * np.nan
        for i in range(raw_vars.shape[1]):
            for j in range(len(self._categories[i])):
                ids = np.where(raw_vars[:, i] == self._categories[i][j])[0]
                transformed_feature[ids, i] = self._cat_features[i][j]
                try:
                    np.isnan(transformed_feature.sum())
                except InputError as err:
                    print('InputError: A wrong experimental variable at ',
                          np.argwhere(np.isnan(transformed_feature)))
        return transformed_feature


class AvgMeasurement(FeatureTransform):
    """ AvgMeasurement class using the average of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self,
                 raw_vars,
                 Y,
                 cat_features=None,
                 categories=None,
                 mode='independent'):
        super(AvgMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories,
            mode=mode)
        self._name = 'AvgMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                if mode == 'independent':
                    choice = list(set(raw_vars[:, i]))
                    categories.append(choice)
                elif mode == 'correlate':
                    choice = list(set(raw_vars.ravel()))
                    categories.append(choice)

        self._categories = categories
        if cat_features == None:
            cat_features = []
            if mode == 'independent':
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = np.where(raw_vars[:, i] == categories[i][j])[0]
                        feature_choice[j] = np.mean(Y[ids])
                    cat_features.append(feature_choice)
            elif mode == 'correlate':
                wild_type = raw_vars[0, :]
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = []
                        for t in range(raw_vars.shape[1]):
                            ids.extend(
                                list(
                                    np.where(
                                        np.logical_and(
                                            raw_vars[:, t] == categories[i][j],
                                            wild_type[t] == wild_type[i]))[0]))
                        feature_choice[j] = np.mean(Y[ids])
                    cat_features.append(feature_choice)

        self._cat_features = cat_features


class RankMeasurement(FeatureTransform):
    """ RankMeasurement class using the rank of the average of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self,
                 raw_vars,
                 Y,
                 cat_features=None,
                 categories=None,
                 mode='independent'):
        super(RankMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories,
            mode=mode)
        self._name = 'RankMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                if mode == 'independent':
                    choice = list(set(raw_vars[:, i]))
                    categories.append(choice)
                elif mode == 'correlate':
                    choice = list(set(raw_vars.ravel()))
                    categories.append(choice)

        self._categories = categories
        if cat_features == None:
            cat_features = []
            if mode == 'independent':
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = np.where(raw_vars[:, i] == categories[i][j])[0]
                        feature_choice[j] = np.mean(Y[ids])
                        ranks = np.argsort(feature_choice)
                    cat_features.append(ranks)
            elif mode == 'correlate':
                wild_type = raw_vars[0, :]
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = []
                        for t in range(raw_vars.shape[1]):
                            ids.extend(
                                list(
                                    np.where(
                                        np.logical_and(
                                            raw_vars[:, t] == categories[i][j],
                                            wild_type[t] == wild_type[i]))[0]))
                        feature_choice[j] = np.mean(Y[ids])
                        ranks = np.argsort(feature_choice)
                    cat_features.append(ranks)
        self._cat_features = cat_features


class MaxMeasurement(FeatureTransform):
    """ MaxMeasurement class using max value of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self,
                 raw_vars,
                 Y,
                 cat_features=None,
                 categories=None,
                 mode='independent'):
        super(MaxMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories,
            mode=mode)
        self._name = 'MaxMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                if mode == 'independent':
                    choice = list(set(raw_vars[:, i]))
                    categories.append(choice)
                elif mode == 'correlate':
                    choice = list(set(raw_vars.ravel()))
                    categories.append(choice)

        self._categories = categories
        if cat_features == None:
            cat_features = []
            if mode == 'independent':
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = np.where(raw_vars[:, i] == categories[i][j])[0]
                        feature_choice[j] = np.max(Y[ids])
                    cat_features.append(feature_choice)
            elif mode == 'correlate':
                wild_type = raw_vars[0, :]
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = []
                        for t in range(raw_vars.shape[1]):
                            ids.extend(
                                list(
                                    np.where(
                                        np.logical_and(
                                            raw_vars[:, t] == categories[i][j],
                                            wild_type[t] == wild_type[i]))[0]))
                        feature_choice[j] = np.max(Y[ids])
                    cat_features.append(feature_choice)
        self._cat_features = cat_features


class MinMeasurement(FeatureTransform):
    """ MinMeasurement class using min value of the measurements 
    of experiments having the apperence of a certain choice for a variable as 
    the feature
    """

    def __init__(self,
                 raw_vars,
                 Y,
                 cat_features=None,
                 categories=None,
                 mode='independent'):
        super(MinMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories,
            mode=mode)
        self._name = 'MinMeasurement'
        if categories == None:
            categories = []
            for i in range(raw_vars.shape[1]):
                if mode == 'independent':
                    choice = list(set(raw_vars[:, i]))
                    categories.append(choice)
                elif mode == 'correlate':
                    choice = list(set(raw_vars.ravel()))
                    categories.append(choice)
        self._categories = categories
        if cat_features == None:
            cat_features = []
            if mode == 'independent':
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = np.where(raw_vars[:, i] == categories[i][j])[0]
                        feature_choice[j] = np.min(Y[ids])
                    cat_features.append(feature_choice)
            elif mode == 'correlate':
                wild_type = raw_vars[0, :]
                for i in range(raw_vars.shape[1]):
                    feature_choice = np.empty(len(categories[i]))
                    for j in range(len(categories[i])):
                        ids = []
                        for t in range(raw_vars.shape[1]):
                            ids.extend(
                                list(
                                    np.where(
                                        np.logical_and(
                                            raw_vars[:, t] == categories[i][j],
                                            wild_type[t] == wild_type[i]))[0]))
                        feature_choice[j] = np.min(Y[ids])
                    cat_features.append(feature_choice)
        self._cat_features = cat_features
