import numpy as np
import torch


def normalize_data(X, Y, X_pending=None):
    train_y = (Y - torch.mean(Y)) / torch.std(Y)
    if X_pending != None:
        X_combine = torch.cat([X, X_pending])
        X_mean, X_std = torch.mean(
            X_combine, dim=0), torch.std(
                X_combine, dim=0)
        X_combine = (X_turbo_combine - X_mean) / X_std
        X_min, X_max = torch.min(
            X_combine, dim=0)[0], torch.max(
                X_combine, dim=0)[0]
        train_x, test_x = (X - X_mean) / X_std, (X_pending - X_mean) / X_std
        train_x, test_x = torch.div(train_x - X_min, X_max - X_min), torch.div(
            test_x - X_min, X_max - X_min)
    else:
        X_mean, X_std = torch.mean(X, dim=0), torch.std(X, dim=0)
        X_min, X_max = torch.min(X, dim=0)[0], torch.max(X, dim=0)[0]
        train_x = (X - X_mean) / X_std
        train_x = torch.div(train_x - X_min, X_max - X_min)
        test_x = None
    stats = [X_mean, X_std, X_min, X_max]
    return train_x, train_y, test_x, stats


def denormalize_X(train_x, stats):
    train_x = torch.multiply(stats[3] - stats[2], train_x) + stats[2]
    X = train_x * stats[1] + stats[0]
    return X


def code_to_array(X):
    name = []
    for i in range(len(X)):
        name.append(list(X[i]))
    name = np.vstack(name)
    return name
