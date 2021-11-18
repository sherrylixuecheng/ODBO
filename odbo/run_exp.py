import numpy as np
import random
import botorch
import torch
from gpytorch.utils.errors import NanError, NotPSDError
from .regressions import GPRegression, RobustRegression
from .utils import normalize_data


def bo_design(X,
              Y,
              X_pending=None,
              gp_method='gp_regression',
              batch_size=1,
              min_inferred_noise_level=1e-4,
              verbose=False):
    from .bo import generate_batch
    X_norm, Y_norm, X_pending_norm, stats = normalize_data(
        X, Y, X_pending=X_pending)

    if gp_method == 'robust_regression':
        model, inliers, outliers = RobustRegression(
            X_norm, Y_norm, min_inferred_noise_level=min_inferred_noise_level)
        X_norm, Y_norm = X_norm[inliers, :], Y_norm[inliers]
        if len(inliers) != len(Y):
            print(len(Y) - len(inliers), ' outliers found')

    while True:
        try:
            gp_model = GPRegression(
                X_norm,
                Y_norm,
                min_inferred_noise_level=min_inferred_noise_level)
            break
        except NotPSDError:
            gp_model = GPRegression(
                X_norm,
                Y_norm,
                min_inferred_noise_level=min_inferred_noise_level * 10,
                optimizer='fit_gpytorch_torch')
            print(
                'The scipy optimizer and minimum inferred noises cannot make the kernel PSD, switch to torch optimizer'
            )
            break

    X_next, acq_value = generate_batch(
        model=gp_model,
        X=X_norm,
        Y=Y_norm,
        batch_size=batch_size,
        X_pending=X_pending_norm)
    next_exp_id = []

    if X_pending is not None:
        for j in range(batch_size):
            tonext = []
            for i in range(X_pending.shape[0]):
                if torch.equal(X_next[j:j + 1, :].detach(),
                               X_pending_norm[i:i + 1, :].detach()):
                    tonext.append(i)
            if len(tonext) > 1: 
                tonext = [random.choice(tonext)]
            next_exp_id.extend(tonext)
            
        if verbose == True:
            print("Next experiment to pick: ",X_next.detach().numpy(),
                  "Acqusition value: ",acq_value.detach().numpy())

    return X_next, acq_value, next_exp_id


def turbo_design(state,
                 X,
                 Y,
                 n_trust_regions=1,
                 X_pending=None,
                 gp_method='gp_regression',
                 batch_size=1,
                 min_inferred_noise_level=1e-4,
                 verbose=False):
    from .turbo import generate_batch
    X_norm, Y_norm, X_pending_norm, stats = normalize_data(
        X, Y, X_pending=X_pending)

    if gp_method == 'robust_regression':
        model, inliers, outliers = RobustRegression(
            X_norm, Y_norm, min_inferred_noise_level=min_inferred_noise_level)
        X_norm, Y_norm = X_norm[inliers, :], Y_norm[inliers]
        if len(inliers) != len(Y):
            print(len(Y) - len(inliers), ' outliers found')

    while True:
        try:
            gp_model = GPRegression(
                X_norm,
                Y_norm,
                min_inferred_noise_level=min_inferred_noise_level)
            break
        except NotPSDError:
            gp_model = GPRegression(
                X_norm,
                Y_norm,
                min_inferred_noise_level=min_inferred_noise_level * 10,
                optimizer='fit_gpytorch_torch')
            print(
                'The scipy optimizer and minimum inferred noises cannot make the kernel PSD, switch to torch optimizer'
            )
            break

    X_next, acq_value = generate_batch(
        state=state,
        model=gp_model,
        X=X_norm,
        Y=Y_norm,
        n_trust_regions=n_trust_regions,
        batch_size=batch_size,
        X_pending=X_pending_norm)
    next_exp_id = []

    if X_pending is not None:
        for t in range(n_trust_regions):
            next_exp_id_m = []
            for i in range(X_pending.shape[0]):
                for j in range(batch_size):
                    if torch.equal(X_next[t, j, :].detach().reshape(1, X_pending_norm.shape[1]),
                                   X_pending_norm[i:i + 1, :].detach()):
                        next_exp_id_m.append(i)
            next_exp_id.append(next_exp_id_m)
        if verbose == True:
            print("Next experiment to pick: ", X_next.detach().numpy(),
                  "Acqusition value: ", acq_value.detach().numpy())
        next_exp_id = np.vstack(next_exp_id)

    return X_next, acq_value, next_exp_id
