import numpy as np
import warnings


class MassiveFeatureTransform(object):
    """MassiveFeatureTransform method
    """

    def __init__(self,
                 raw_vars,
                 cat_features=None,
                 Y=None,
                 categories=None,
                 method='Avg',
                 mode='independent'):
        """Constructor for the MassiveFeatureTransform class.
        Args:
            raw_vars : Input experiments expressed using raw variable names
            cat_features : Featurizations of different categories of each variable
            Y : Input training measurements
            categories : Categories/Choices of each variable
            method : Method to use the measurements as features, must be 'Avg', 'Max',
                     'Min', 'Rank'
            mode : Mode to infer features at different varible locations. 'independent' 
                   means varibles independently vary, 'correlate' means all the variables 
                   share the same features if the experimental choices are the same
        """

        self._raw_vars = raw_vars
        self._cat_features = cat_features
        self._Y = Y
        self._catergories = categories
        self._mode = mode
        self._method = method

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
                        if self._method == 'Avg':
                            feature_choice[j] = np.mean(Y[ids])
                        elif self._method == 'Max':
                            feature_choice[j] = np.max(Y[ids])
                        elif self._method == 'Min':
                            feature_choice[j] = np.min(Y[ids])
                        elif self._method == 'Rank':
                            feature_choice[j] = np.mean(Y[ids])
                            feature_choice = np.argsort(feature_choice)
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
                        if self._method == 'Avg':
                            feature_choice[j] = np.mean(Y[ids])
                        elif self._method == 'Max':
                            feature_choice[j] = np.max(Y[ids])
                        elif self._method == 'Min':
                            feature_choice[j] = np.min(Y[ids])
                        elif self._method == 'Rank':
                            feature_choice[j] = np.mean(Y[ids])
                            feature_choice = np.argsort(feature_choice)
                    cat_features.append(feature_choice)

        self._cat_features = cat_features

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


class FewChangeMeasurement(MassiveFeatureTransform):
    """FewChangeMeasurement method
    """

    def __init__(self,
                 raw_vars,
                 cat_features=None,
                 Y=None,
                 categories=None,
                 max_change_length=None,
                 n_components=None,
                 method='Avg',
                 mode='independent',
                 random_seed=0):
        """Constructor for the FewChangeMeasurement class.
        Args:
            raw_vars : Input experiments expressed using raw variable names
            cat_features : Featurizations of different categories of each variable
            Y : Input training measurements
            categories : Categories/Choices of each variable
            max_change_length : Maximum numbers of varibles changing in an experiment
            method : Method to use the measurements as features, must be 'Avg', 'Max',
                     'Min', 'Rank'
            mode : Mode to infer features at different varible locations. 'independent' 
                   means varibles independently vary, 'correlate' means all the variables 
                   share the same features if the experimental choices are the same
        """
        super(FewChangeMeasurement, self).__init__(
            raw_vars=raw_vars,
            cat_features=cat_features,
            Y=Y,
            categories=categories,
            mode=mode,
            method=method)
        self._max_change_length = max_change_length
        self._pca = None
        self._random_seed = random_seed
        self._n_components = n_components

        if self._max_change_length == None:
            self._max_change_length = 0
            for i in range(1, raw_vars.shape[0]):
                curr_len = len(np.where(raw_vars[0, :] != raw_vars[i, :])[0])
                if curr_len >= self._max_change_length:
                    self._max_change_length = curr_len

    def transform(self, raw_vars):
        """Transform input experiments to standard encodings
        Args:
            raw_vars : Input experiments expressed using raw variable names        
        """
        for i in range(1, raw_vars.shape[0]):
            curr_len = len(np.where(raw_vars[0, :] != raw_vars[i, :])[0])
            if curr_len >= self._max_change_length:
                warnings.warn(
                    "Entire search space changes more varibles per experiment. Need to retransform the training space"
                )
                self._max_change_length = curr_len
                self._pca = None

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
        if self._n_components is None:
            self._n_components = 4 * self._max_change_length
        if self._pca is None:
            from sklearn.decomposition import PCA
            self._pca = PCA(
                n_components=self._n_components, random_state=self._random_seed)
            self._pca.fit(transformed_feature)
        transformed_feature_pca = self._pca.transform(transformed_feature)
        return transformed_feature, transformed_feature_pca
