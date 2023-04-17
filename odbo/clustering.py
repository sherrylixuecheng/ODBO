"""Clustering algorithms with automatic cluster number detection"""

import numpy as np


def gmm(ncluster_grid, X, random_state=0, **kwargs):
    """GMM clustering with full covariance computation using BIC score as auto model selection
    Parameters
    ----------
    ncluster_grid : list of ints
        List of possible number of clusters. 
    X : ndarray with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation.  
    random_state : int, default=0
        Random seed for GMM clustering.
    **kwargs : options
        Additional options in sklearn.
    Returns
    -------
    best_gmm : Sklearn GMM model
        The best selected GMM model using BIC score 
    bic : list of floats
        BIC scores for all the possible number of clusters.
    Note
    -------
    More options see: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    References
    ----------
    Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    from sklearn import mixture
    bic = []
    lowest_bic = np.infty
    for i in ncluster_grid:
        if i <= X.shape[0]:
            gmm_model = mixture.GaussianMixture(n_components=i,
                                                covariance_type='full',
                                                random_state=random_state,
                                                **kwargs)
            gmm_model.fit(X)
            bic.append(gmm_model.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm_model
    return best_gmm, bic


def kmeans(ncluster_grid, X, random_state=0):
    """KMeans clustering with full covariance computation using Davies-Bouldin score as auto model selection
    Parameters
    ----------
    ncluster_grid : list of ints
        List of possible number of clusters. 
    X : ndarray with a shape of (n_training_samples, feature_size) of floats
        Current set of experimental design after featurziation.  
    random_state : int, default=0
        Random seed for KMeans clustering.
    **kwargs : options
        Additional options in sklearn.
    Returns
    -------
    best_kmeans : Sklearn GMM model
        The best selected KMeans model using BIC score 
    db_score : list of floats
        Davies-Bouldin scores for all the possible number of clusters.
    Note
    -------
    More options see: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    References
    ----------
    Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    """
    from sklearn import cluster

    db_score = []
    lowest_db_score = np.infty
    for i in ncluster_grid:
        kmeans_model = cluster.KMeans(n_components=i,
                                      random_state=random_state)
        kmeans_model.fit(X)
        db_score.append(
            sklearn.metrics.davies_bouldin_score(X,
                                                 labels=kmeans_model.labels_))
        if db_score[-1] < lowest_db_score:
            lowest_db_score = db_score[-1]
            best_kmeans = kmeans_model
    return best_kmeans, db_score
