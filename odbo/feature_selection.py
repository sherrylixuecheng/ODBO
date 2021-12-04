from typing import List, Tuple, Optional, Union
import numpy as np

def rf_feature_selection(
    X: np.ndarray,
    Y: np.ndarray,
    threshold: float = 1e-4,
    n_splits: int = 20,
    test_size: float = 0.5,
    impurity: bool = False,
    feature_sele_random: Optional[int] = None,
) -> Tuple[List[int], List[float], Union[float, None]]:
    """Selects features based on Random Forest Regression with mean decrease of accuracy criteria.

    Parameters
    ----------
    X : ndarray
        The input features, rows are pairs and columns are features
    Y : ndarray
        The input pair energies
    threshold : float, optional (default 1e-4)
        Importance threshold to select the features
    n_splits : int, optional (default 20)
        Number of trials in the RFR
    test_size : float, optional (default 0.5)
        Represents the proportion of the dataset to include in the test split in each RFR trial
    impurity : boolean, optional (default False)
        Perform IPR impurity calculation
    feature_sele_random : int, optional (default None)
        Generate random number from the random seed.

    Returns
    -------
    selected_features : list of int
        List of selected features
    importance : list of floats
        Importance of the selected features
    IPR : float or None
        If float, the IPR calculation is performed and the impurity is returned
        If None, the IPR calculation is not performed

    Raises
    ------
    ImportError
        If scikit-learn (sklearn) cannot be found
    """
    from sklearn.model_selection import ShuffleSplit
    from sklearn.ensemble import RandomForestRegressor
    from collections import defaultdict
    from sklearn.metrics import r2_score

    rf = RandomForestRegressor()
    scores = defaultdict(list)
    if feature_sele_random is None:
        test_train_split = ShuffleSplit(n_splits=n_splits, test_size=test_size)
    elif isinstance(feature_sele_random, int):
        test_train_split = ShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=feature_sele_random)
    else:
        raise Exception("The specified feature_sele_random needs to be an integer.")

    for train_idx, test_idx in test_train_split.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx].ravel(), Y[test_idx].ravel()
        rf.fit(X_train, Y_train)
        accuracy = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test
            col_before_shuffle = X_t[:, i].copy()
            np.random.shuffle(X_t[:, i])
            shuffle_accuracy = r2_score(Y_test, rf.predict(X_t))
            X_t[:, i] = col_before_shuffle.copy()
            scores[i].append((accuracy - shuffle_accuracy) / accuracy)

    sorted_scores = sorted(
        [(np.mean(feature_score), feature_index) for feature_index, feature_score in scores.items()],
        reverse=True)

    k = 0
    selected_features = []
    importance = []
    while sorted_scores[k][0] >= threshold:
        selected_features.append(int(sorted_scores[k][1]))
        importance.append(sorted_scores[k][0])
        k = k + 1

    IPR = None
    if impurity:
        renorm = importance / np.sum(importance)
        IPR = 1 / np.sum([i**2 for i in renorm])

    return selected_features, importance, IPR
