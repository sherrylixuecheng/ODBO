"""Initialization algorithm to find the 0th round experiments for BO"""

import numpy as np


def initial_design(X_pending,
                   choice_list=None,
                   least_occurance=None,
                   target_num=None,
                   importance_method='sum',
                   verbose=True,
                   allow_abundance=False,
                   random_state=0):
    np.random.seed(random_state)
    if least_occurance == None:
        least_occurance = np.ones(X_pending.shape[1])
    if choice_list == None:
        choice_list = []
        for i in range(X_pending.shape[1]):
            choice_list.append(list(set(X_pending[:, i])))
    N = X_pending.shape[0]
    if allow_abundance:
        abundance_scores = abundance(X_pending, choice_list)
    pending_scores = np.zeros(X_pending.shape)
    sele_indices = [0]
    pending_indices = np.delete(range(N), sele_indices)
    pending_scores[sele_indices, :] = -np.inf * np.ones(
        pending_scores[sele_indices, :].shape)
    pending_scores[pending_indices, :] = compute_score(
        X_pending[sele_indices, :], X_pending[pending_indices, :],
        least_occurance, choice_list)
    if verbose == True:
        print('Current selected experiments: ', sele_indices[-1],
              'Max pending score: ', np.max(pending_scores))

    while True:
        if allow_abundance:
            sum_scores = np.sum(
                np.multiply(pending_scores, abundance_scores), axis=1)
        else:
            sum_scores = np.sum(pending_scores, axis=1)
        sum_scores[sele_indices] = -np.inf * np.ones(len(sele_indices))
        if np.max(pending_scores) <= 0.0:
            break
        update_indices = np.argmax(sum_scores)
        sele_indices.append(update_indices)
        pending_scores = update_score(pending_scores,
                                      X_pending[update_indices, :], X_pending)
        if verbose == True:
            print('Current selected experiments: ', sele_indices[-1],
                  'Max pending score: ', np.max(pending_scores))
    return sele_indices


def compute_score(current_X,
                  X_pending,
                  least_occurance,
                  choice_list,
                  importance_method='sum'):
    scores = np.zeros(X_pending.shape)
    if importance_method == 'sum':
        for i in range(X_pending.shape[1]):
            for j in range(len(choice_list[i])):
                current_id_no = np.where(
                    current_X[:, i] == choice_list[i][j])[0]
                pending_id_no = np.where(
                    X_pending[:, i] == choice_list[i][j])[0]
                raw_score = least_occurance[i] - len(current_id_no)
                scores[pending_id_no, i] = raw_score
        return scores
    else:
        print(
            'Other importance score computation method is not implemented yet.'
        )


def update_score(pending_scores, update_X, X_pending):
    for i in range(X_pending.shape[1]):
        pending_id_no = np.where(X_pending[:, i] == update_X[i])[0]
        pending_scores[pending_id_no, i] = pending_scores[pending_id_no, i] - 1
    return pending_scores


def abundance(X_pending, choice_list):
    N, feature_size = X_pending.shape[0], X_pending.shape[1]
    abundance = np.zeros(X_pending.shape)
    for i in range(feature_size):
        for j in range(len(choice_list[i])):
            pending_id_no = np.where(X_pending[:, i] == choice_list[i][j])[0]
            abundance[pending_id_no, i] = (
                N / len(choice_list[i]) / len(pending_id_no))**2 * np.ones(
                    len(pending_id_no))
    return abundance
